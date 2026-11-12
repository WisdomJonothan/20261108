#!/usr/bin/env python3
"""Keep only the given qids of a TREC run (used to build a val-only candidate pool)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib_reteco import read_trec_run, write_trec_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qids", required=True, help="json: {domain: [qid, ...]} or a flat list")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.qids, encoding="utf-8") as f:
        spec = json.load(f)
    want = spec.get("eval", spec)
    qids = want.get(args.domain, []) if isinstance(want, dict) else want
    run = read_trec_run(args.run)
    keep = {q: [(doc, s) for _, s, doc in run[q]] for q in qids if q in run}
    write_trec_run(args.out, keep)
    print(f"{args.domain}: kept {len(keep)}/{len(qids)} qids -> {args.out}")


if __name__ == "__main__":
    main()
