#!/usr/bin/env python3
"""Is a deeper candidate pool worth the cross-encoder time?

Recall inside the dense top-100 is 0.6995 (domain-macro); inside the top-1000 it
is 0.9247.  This script answers the only question that matters for spending 3x
more reranker compute: does re-scoring a deeper pool actually raise nDCG@10 once
the same CE re-ranks it, or does the CE's precision collapse at depth?

Step 1 (--select) picks a dev query sample with genuine depth headroom
(0 < Rec@100 < 1, gold present within the dense top-500), writes
examples_depth_<domain>.jsonl for run_chunk_rerank.py, and records the ids.
Step 2 (--score) reads the resulting 200-deep CE runs and compares, on exactly
the same queries, the fused system at pool 100 vs pool 200 (z-sum 1:1).

  python scripts/depth_probe.py --select
  # then, per domain in results/depth_probe/selection.json:
  #   python scripts/run_chunk_rerank.py --domain D --split dev \
  #     --examples results/depth_probe/examples_depth_D.jsonl --k 200 \
  #     --model results/ce_ft/v2m3_pool8_e1/best --q-mode title --q-cap 256 \
  #     --win 250 --stride 192 --pool max --batch 32 \
  #     --out runs/rerank_depth_D_dev.trec
  python scripts/depth_probe.py --score
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_qrels, read_jsonl, read_trec_run  # noqa: E402

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]
DEPTH = 200
OUT = "results/depth_probe"


def ndcg10(ranked, gold):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg else 0.0


def select(root, per_domain_cap=6):
    os.makedirs(os.path.join(root, OUT), exist_ok=True)
    sel, total = {}, 0
    for d in DOMAINS:
        ddir = os.path.join(root, "data", "track1_tempo", d)
        gold = {q: set(v) for q, v in load_qrels(ddir, "dev").items()}
        dense = read_trec_run(os.path.join(root, f"runs/dense_mxbai_{d}_dev.trec"))
        recs = {r["id"]: r for r in read_jsonl(os.path.join(ddir, "examples_dev.jsonl"))}
        picks = []
        for q, gl in gold.items():
            ranked = [doc for _r, _s, doc in dense.get(q, [])]
            if not ranked or not gl:
                continue
            r100 = len(set(ranked[:100]) & gl) / len(gl)
            r500 = len(set(ranked[:500]) & gl) / len(gl)
            if 0 < r100 < 1 and r500 > r100:
                picks.append({"qid": q, "rec100": r100, "rec500": r500})
        # hardest first: most gold still hidden at depth
        picks.sort(key=lambda x: (x["rec100"], -x["rec500"]))
        take = picks[:per_domain_cap]
        if not take:
            continue
        path = os.path.join(root, OUT, f"examples_depth_{d}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for p in take:
                fh.write(json.dumps({"id": p["qid"], "query": recs[p["qid"]]["query"]},
                                    ensure_ascii=False) + "\n")
        sel[d] = take
        total += len(take)
        print(f"{d:10s} {len(take):2d} queries  (pool of {len(picks)})")
    with open(os.path.join(root, OUT, "selection.json"), "w", encoding="utf-8") as fh:
        json.dump(sel, fh, indent=1)
    print(f"selected {total} queries")


def fuse(dense, ce, k, w=1.0):
    docs = [d for d in dense if d in ce][:k]
    if not docs:
        return []
    a = np.asarray([dense[d] for d in docs], float)
    b = np.asarray([ce[d] for d in docs], float)
    a = (a - a.mean()) / a.std() if a.std() > 0 else np.zeros_like(a)
    b = (b - b.mean()) / b.std() if b.std() > 0 else np.zeros_like(b)
    s = a + w * b
    return [docs[i] for i in np.argsort(-s, kind="stable")]


def score(root, boot=4000, seed=7):
    sel = json.load(open(os.path.join(root, OUT, "selection.json"), encoding="utf-8"))
    per_domain = {}
    rows = []
    for d, picks in sel.items():
        ddir = os.path.join(root, "data", "track1_tempo", d)
        gold = {q: set(v) for q, v in load_qrels(ddir, "dev").items()}
        dense = read_trec_run(os.path.join(root, f"runs/dense_mxbai_{d}_dev.trec"))
        ce = read_trec_run(os.path.join(root, f"runs/rerank_depth_{d}_dev.trec"))
        vals = {100: [], 200: []}
        for p in picks:
            q = p["qid"]
            dd = {doc: sc for _r, sc, doc in dense.get(q, [])}
            cc = {doc: sc for _r, sc, doc in ce.get(q, [])}
            for k in vals:
                vals[k].append(ndcg10(fuse(dd, cc, k), gold[q]))
        diff = np.array(vals[200]) - np.array(vals[100])
        per_domain[d] = {"n": len(diff), "k100": float(np.mean(vals[100])),
                         "k200": float(np.mean(vals[200])), "delta": float(diff.mean())}
        rows.extend(zip(vals[100], vals[200]))
        print(f"{d:10s} n={len(diff):2d}  k100 {per_domain[d]['k100']:.4f}  "
              f"k200 {per_domain[d]['k200']:.4f}  delta {per_domain[d]['delta']:+.4f}")
    deltas = np.array([b - a for a, b in rows])
    rng = np.random.default_rng(seed)
    boots = [deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\noverall n={len(deltas)} delta {deltas.mean():+.4f}  "
          f"95% query-bootstrap [{lo:+.4f},{hi:+.4f}]  "
          f"queries up {int((deltas > 0).sum())} / down {int((deltas < 0).sum())}")
    with open(os.path.join(root, OUT, "probe_result.json"), "w", encoding="utf-8") as fh:
        json.dump({"depth": DEPTH, "per_domain": per_domain,
                   "overall": {"n": int(len(deltas)), "delta": float(deltas.mean()),
                               "ci": [float(lo), float(hi)],
                               "up": int((deltas > 0).sum()), "down": int((deltas < 0).sum())}},
                  fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--cap", type=int, default=6)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.select:
        select(root, args.cap)
    elif args.score:
        score(root)
    else:
        ap.error("pass --select or --score")


if __name__ == "__main__":
    main()
