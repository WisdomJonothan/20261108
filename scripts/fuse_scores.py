#!/usr/bin/env python3
"""Score-level fusion of a dense run and a cross-encoder run.

RRF throws the score magnitudes away and merges by rank. When one component is
clearly stronger than the other (here: dense 0.294 vs zero-shot CE 0.268) RRF
taxes the strong one. This fuses on **z-normalised scores** instead, so the
weight has a meaning and the stronger component can dominate:

    s(d) = wa * z(score_dense(d)) + wb * z(score_ce(d))

`z` is computed **within a query** over the intersection of the two runs (which
is the frozen candidate list, so recall is unchanged). `--norm none` gives the
plain weighted score sum; `--norm minmax` maps each query's scores to [0,1].

No qrels are read.

Usage:
  python fuse_scores.py --run-a runs/dense_mxbai_bitcoin_dev.trec \
      --run-b runs/rerank_mxbai_bitcoin_dev.trec \
      --out runs/scoresum_1_1_bitcoin_dev.trec --json results/scoresum_1_1_bitcoin_dev.json \
      --wa 1 --wb 1 --norm z
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import read_trec_run, write_trec_run  # noqa: E402


def norm(values, mode):
    v = np.asarray(values, dtype=float)
    if mode == "none":
        return v
    if mode == "minmax":
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    mu, sd = v.mean(), v.std()
    return (v - mu) / sd if sd > 0 else np.zeros_like(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="dense run (defines the candidate set)")
    ap.add_argument("--run-b", required=True, help="cross-encoder run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--norm", choices=["z", "minmax", "none"], default="z")
    ap.add_argument("--tag", default="scoresum")
    args = ap.parse_args()

    A = read_trec_run(args.run_a)
    B = read_trec_run(args.run_b)
    fused = {}
    stats = {"queries": 0, "common": 0, "a_only": 0, "b_only": 0, "all_tied": 0}
    for qid, rows in A.items():
        sa = {t[2]: t[1] for t in rows}
        sb = {t[2]: t[1] for t in B.get(qid, [])}
        common = [d for d in sa if d in sb]
        stats["queries"] += 1
        stats["common"] += len(common)
        stats["a_only"] += len(sa) - len(common)
        stats["b_only"] += len(sb) - len(common)
        if not common:
            fused[qid] = [(t[2], float(t[1])) for t in rows]
            continue
        za = norm([sa[d] for d in common], args.norm)
        zb = norm([sb[d] for d in common], args.norm)
        book = dict(zip(common, args.wa * za + args.wb * zb))
        # union: keep the CE-only tail, ordered by its normalised score
        order = sorted(book, key=lambda d: -book[d])
        if len(set(round(book[d], 12) for d in order)) == 1:
            stats["all_tied"] += 1
        fused[qid] = [(d, float(book[d])) for d in order]
        for d in sa:
            if d not in book:
                fused[qid].append((d, float("-1e9") + sa[d] * 1e-6))
    write_trec_run(args.out, fused, tag=args.tag)
    meta = {"run_a": os.path.abspath(args.run_a), "run_b": os.path.abspath(args.run_b),
            "wa": args.wa, "wb": args.wb, "norm": args.norm, **stats,
            "note": "scores are fused values, not comparable across queries"}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("queries", "common", "a_only", "b_only", "all_tied")}))


if __name__ == "__main__":
    main()
