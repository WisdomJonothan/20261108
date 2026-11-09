#!/usr/bin/env python3
"""Filter a TREC run down to a query subset (used for the held-out val45 fit).

Usage:
  python make_subset_run.py --run runs/dense_mxbai_bitcoin_train.trec \
      --examples results/ce_val45/examples_val45_bitcoin.jsonl \
      --out runs/subset_val45_bitcoin_train.trec
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import read_trec_run, write_trec_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    keep = []
    with open(a.examples, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keep.append(json.loads(line)["id"])
    run = read_trec_run(a.run)
    missing = [q for q in keep if q not in run]
    out = {q: [(d, s) for _r, s, d in run[q]] for q in keep if q in run}
    write_trec_run(a.out, out, tag="subset")
    print(json.dumps({"kept": len(out), "requested": len(keep),
                      "missing": missing, "out": a.out}))


if __name__ == "__main__":
    main()
