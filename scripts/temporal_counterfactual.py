#!/usr/bin/env python3
"""Counterfactual temporal reranking probe (TACR).

For every year mentioned in a candidate we build three edits of that candidate:

  aligned : years moved to the nearest query anchor year
  neutral : years moved by +/-1 (keeps the year inside the anchor tolerance)
  wrong   : years moved far away from every anchor

Everything else in the document is untouched, so score differences come from the
temporal span only. The probe answers two questions:

  1. Does the cross-encoder react to the year at all?
     effect = score(aligned) - score(wrong)
  2. Can that reaction be turned into a ranking feature?
     feature = score(aligned) - score(original)   (0 for docs without years)
     and we measure gold-vs-non-gold AUC of the feature and the nDCG@10 delta of
     re-scoring candidates as score + lambda * feature (lambda fixed a priori).

Dev queries only, dense candidates only, qrels only for evaluation.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_qrels, read_jsonl, read_trec_run  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def years_in(text):
    return sorted({int(y) for y in YEAR.findall(text or "")})


def query_anchors(query, guid):
    parts = [query]
    if guid:
        for k in ("query_summary", "query_temporal_events", "key_time_anchors"):
            v = guid.get(k)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts.extend(str(x) for x in v)
    return years_in("\n".join(parts))


def sub_years(text, fn):
    return YEAR.sub(lambda m: str(fn(int(m.group(0)))), text)


def make_variants(text, anchors, tol=1, far=6):
    ys = years_in(text)
    if not ys or not anchors:
        return None
    near = lambda y: min(anchors, key=lambda a: abs(a - y))
    aligned = sub_years(text, lambda y: near(y))
    if aligned == text:
        return None
    neutral = None
    for d in (1, -1):
        cand = [y + d for y in ys]
        if all(min(abs(c - a) for a in anchors) <= tol for c in cand):
            neutral = sub_years(text, lambda y, d=d: y + d)
            break
    if neutral is None:
        neutral = aligned
    wrong, shift = None, None
    for d in (far, -far, far + 2, -(far + 2)):
        cand = [y + d for y in ys]
        if all(min(abs(c - a) for a in anchors) > tol for c in cand) and all(1000 <= c <= 2099 for c in cand):
            wrong, shift = sub_years(text, lambda y, d=d: y + d), d
            break
    if wrong is None:
        return None
    return {"aligned": aligned, "neutral": neutral, "wrong": wrong, "shift_far": shift,
            "doc_years": ys}


def auc(scores, labels):
    """Mann-Whitney AUC; ties count 1/2."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ndcg(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--q-cap", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lambdas", default="0.25,0.5,1.0")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "bge-reranker-v2-m3"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tacr] model={args.model} device={device}", flush=True)
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
    run = read_trec_run(args.run)
    if os.path.isfile(os.path.join(ddir, f"qrels_{args.split}.txt")):
        gold_all = load_qrels(ddir, args.split)
    else:
        gold_all = {q: set(m["gold_ids"]) for q, m in qmeta.items()}

    def score(pairs):
        out = []
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i:i + args.batch_size]
            enc = tok([p[0] for p in batch], [p[1] for p in batch], padding=True,
                      truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                out.extend(model(**enc).logits.view(-1).float().cpu().tolist())
        return out

    lambdas = [float(x) for x in args.lambdas.split(",")]
    rows_all, per_query = [], {}
    for qid, ranked in run.items():
        if qid not in qmeta:
            continue
        anchors = query_anchors(qmeta[qid]["query"], guid.get(qid))
        if not anchors:
            continue
        qt = re.sub(r"<[^>]+>", " ", qmeta[qid]["query"])[:args.q_cap]
        cands = []
        for _r, _s, did in ranked[:args.k]:
            v = make_variants(doc.get(did, ""), anchors)
            if v:
                cands.append((did, v))
        if len(cands) < 5:
            continue
        pairs = []
        for did, v in cands:
            pairs += [(qt, doc[did][:4000]), (qt, v["aligned"][:4000]),
                      (qt, v["neutral"][:4000]), (qt, v["wrong"][:4000])]
        sc = score(pairs)
        gold = set(gold_all.get(qid, []))
        recs = []
        for j, (did, v) in enumerate(cands):
            s0, sa, sn, sw = sc[4 * j], sc[4 * j + 1], sc[4 * j + 2], sc[4 * j + 3]
            recs.append({"doc": did, "s0": s0, "sa": sa, "sn": sn, "sw": sw,
                         "feat": sa - s0, "effect": sa - sw, "gold": did in gold})
        rows_all += [dict(qid=qid, **r) for r in recs]
        per_query[qid] = {"n": len(recs), "anchors": anchors,
                          "auc_feat": auc([r["feat"] for r in recs], [r["gold"] for r in recs]),
                          "auc_effect": auc([r["effect"] for r in recs], [r["gold"] for r in recs])}

    S = np.array([[r["s0"], r["sa"], r["sn"], r["sw"], r["feat"], r["effect"], r["gold"]]
                  for r in rows_all], dtype=float)
    s0, sa, sn, sw, feat, effect, gold = S.T
    summary = {
        "domain": args.domain, "split": args.split, "k": args.k,
        "queries": len(per_query), "pairs": len(rows_all),
        "mean_score": {"orig": float(s0.mean()), "aligned": float(sa.mean()),
                       "neutral": float(sn.mean()), "wrong": float(sw.mean())},
        "mean_effect_aligned_minus_wrong": float(effect.mean()),
        "auc_feature": float(auc(feat, gold)),
        "auc_effects": float(auc(effect, gold)),
        "per_query_auc_feature_mean": float(np.nanmean([v["auc_feat"] for v in per_query.values()])),
        "auc_s0": float(auc(s0, gold)),
    }
    # simulated rerank inside the candidate window (top-50 -> top-10)
    base, res = [], {l: [] for l in lambdas}
    for qid, grp in _group(rows_all):
        gold_q = set(gold_all.get(qid, []))
        order0 = [r["doc"] for r in sorted(grp, key=lambda r: -r["s0"])]
        base.append(ndcg(order0, gold_q))
        for l in lambdas:
            order = [r["doc"] for r in sorted(grp, key=lambda r: -(r["s0"] + l * r["feat"]))]
            res[l].append(ndcg(order, gold_q))
    summary["ndcg10_candidates_only"] = {"orig": float(np.mean(base)),
                                         **{f"lambda_{l}": float(np.mean(v)) for l, v in res.items()}}
    summary["ndcg10_delta"] = {f"lambda_{l}": float(np.mean(v) - np.mean(base)) for l, v in res.items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"summary": summary, "per_query": per_query,
               "pairs_sample": rows_all[:300]},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def _group(rows):
    g = {}
    for r in rows:
        g.setdefault(r["qid"], []).append(r)
    return sorted(g.items())


if __name__ == "__main__":
    main()
