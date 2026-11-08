#!/usr/bin/env python3
"""TACR rerank test: does counterfactual year alignment improve ranking?

Pipeline per query (dev only):
  1. take the dense top-100 run (frozen, no labels),
  2. score every candidate with the cross-encoder            -> s0,
  3. for candidates that mention a year, score two edits:
        aligned : years moved onto the query anchor years   -> sa
        wrong   : years moved far away from every anchor    -> sw
     feature = sa - s0  (only defined for year-bearing docs, else 0),
  4. rank by  s0 + lambda * feature   and by  s0 - lambda * (s0 - sw),
  5. compare nDCG@10 against the dense order and the plain-CE order.

This tests whether the cross-encoder's year sensitivity (measured in the probe)
can be turned into a ranking gain, on domains where years are actually present.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_qrels, read_jsonl, read_trec_run  # noqa: E402
from temporal_counterfactual import make_variants, query_anchors  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ndcg(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--q-cap", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lambdas", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "bge-reranker-v2-m3"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tacr-rr] {args.domain} model={args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    ddir = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    ids, texts = load_corpus(ddir)
    doc = dict(zip(ids, texts))
    qmeta = {r["id"]: r for r in read_jsonl(os.path.join(ddir, f"examples_{args.split}.jsonl"))}
    gpath = os.path.join(ddir, f"guidance_{args.split}.jsonl")
    guid = {r["id"]: r.get("query_guidance", {}) for r in read_jsonl(gpath)} if os.path.isfile(gpath) else {}
    qrels = load_qrels(ddir, args.split)
    run = read_trec_run(args.run)

    def score(pairs):
        out = []
        for i in range(0, len(pairs), args.batch_size):
            b = pairs[i:i + args.batch_size]
            enc = tok([p[0] for p in b], [p[1] for p in b], padding=True, truncation=True,
                      max_length=args.max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                out.extend(model(**enc).logits.view(-1).float().cpu().tolist())
        return out

    lambdas = [float(x) for x in args.lambdas.split(",")]
    rows = {}
    for qid, ranked in run.items():
        if qid not in qmeta or qid not in qrels:
            continue
        anchors = query_anchors(qmeta[qid]["query"], guid.get(qid))
        docs = [(did, i, s) for i, (_r, s, did) in enumerate(ranked[: args.k])]
        qt = qmeta[qid]["query"]
        pairs, layout = [], []
        for did, _i, _s in docs:
            t = doc.get(did, "")[:4000]
            pairs.append((qt[:args.q_cap], t))
            layout.append((did, "orig"))
            v = make_variants(t, anchors) if anchors else None
            if v:
                pairs.append((qt[:args.q_cap], v["aligned"][:4000])); layout.append((did, "aligned"))
                pairs.append((qt[:args.q_cap], v["wrong"][:4000])); layout.append((did, "wrong"))
        sc = score(pairs)
        s0, sa, sw = {}, {}, {}
        for (did, kind), s in zip(layout, sc):
            {"orig": s0, "aligned": sa, "wrong": sw}[kind][did] = s
        feats = {d: sa[d] - s0[d] for d in sa}
        rows[qid] = {"docs": [d for d, _, _ in docs], "s0": s0, "sa": sa, "sw": sw,
                     "feat": feats, "anchors": anchors}
        print(f"  {qid}: {len(docs)} cands, {len(feats)} with year", flush=True)

    def order(qid, key):
        r = rows[qid]
        return sorted(r["docs"], key=lambda d: -key(d))

    base = {q: ndcg([did for _r, _s, did in run[q][:args.k]], qrels[q]) for q in rows}
    ce = {}
    for q in rows:
        s0 = rows[q]["s0"]
        ce[q] = ndcg(sorted(rows[q]["docs"], key=lambda d, s0=s0: -s0.get(d, -1e9)), qrels[q])
    res = {}
    for lam in lambdas:
        for name in ("feat", "penalty"):
            acc = {}
            for q in rows:
                s0, feat, sw = rows[q]["s0"], rows[q]["feat"], rows[q]["sw"]
                if name == "feat":
                    key = (lambda d, s0=s0, feat=feat, lam=lam:
                           s0.get(d, -1e9) + lam * feat.get(d, 0.0))
                else:
                    key = (lambda d, s0=s0, sw=sw, lam=lam:
                           s0.get(d, -1e9) - lam * max(0.0, s0.get(d, 0.0) - sw.get(d, s0.get(d, 0.0))))
                acc[q] = ndcg(sorted(rows[q]["docs"], key=key), qrels[q])
            res[f"{name}_{lam}"] = acc

    def boot(a, b, n=10000, seed=0):
        rng = np.random.default_rng(seed)
        qs = sorted(a)
        d = np.array([a[q] - b[q] for q in qs])
        idx = rng.integers(0, len(d), (n, len(d)))
        means = d[idx].mean(1)
        return {"mean": float(d.mean()), "ci": [float(np.percentile(means, 2.5)),
                                                float(np.percentile(means, 97.5))],
                "p_gt0": float((means > 0).mean()), "n": len(d)}

    out = {"domain": args.domain, "split": args.split, "k": args.k,
           "queries": len(rows),
           "macro": {"dense": float(np.mean(list(base.values()))),
                     "plain_ce": float(np.mean(list(ce.values()))),
                     **{k: float(np.mean(list(v.values()))) for k, v in res.items()}},
           "vs_plain_ce": {k: boot(v, ce) for k, v in res.items()},
           "vs_dense": {k: boot(v, base) for k, v in res.items()},
           "ce_vs_dense": boot(ce, base)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
