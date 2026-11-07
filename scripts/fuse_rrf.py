"""Reciprocal-rank fusion of two TREC runs over the same candidate set.

score(d) = w_a / (c + rank_a(d)) + w_b / (c + rank_b(d))

The output is restricted to the intersection of the two runs' document sets
(here: the frozen dense candidate list), so recall is unchanged.

Usage:
  python fuse_rrf.py --run-a dense.trec --run-b ce.trec --out fused.trec \
      --json meta.json --wa 2 --wb 1 --c 60
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import read_trec_run, write_trec_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--wa", type=float, default=2.0)
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=60.0)
    args = ap.parse_args()

    A = read_trec_run(args.run_a)
    B = read_trec_run(args.run_b)
    fused, stats = {}, {"queries": 0, "intersection": 0, "a_only": 0, "b_only": 0}
    for qid in A:
        a = [t[2] for t in A[qid]]
        b = [t[2] for t in B.get(qid, [])]
        sa = {d: i + 1 for i, d in enumerate(a)}
        sb = {d: i + 1 for i, d in enumerate(b)}
        score = {}
        for d in sa:
            s = args.wa / (args.c + sa[d])
            if d in sb:
                s += args.wb / (args.c + sb[d])
                stats["intersection"] += 1
            else:
                stats["a_only"] += 1
            score[d] = s
        for d in sb:
            if d not in sa:
                stats["b_only"] += 1
                score[d] = args.wb / (args.c + sb[d])
        ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
        fused[qid] = ranked
        stats["queries"] += 1

    write_trec_run(args.out, fused, tag="rrf")
    meta = {"run_a": os.path.abspath(args.run_a), "run_b": os.path.abspath(args.run_b),
            "wa": args.wa, "wb": args.wb, "c": args.c,
            "stats": stats, "run_file": os.path.abspath(args.out)}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"fused {stats['queries']} queries; intersection {stats['intersection']}")


if __name__ == "__main__":
    main()
