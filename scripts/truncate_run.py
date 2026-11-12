#!/usr/bin/env python
"""Truncate a TREC run to the top K rows per topic.

Used for the depth control in results/MULTIEVIDENCE_NOTE.md: truncating a fused
run from 1000 to 100 rows leaves its top ten unchanged, so step coverage at 10
must stay the same while step recall drops.
"""
import argparse
import collections

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="glob pattern for input TREC files")
ap.add_argument("--out", required=True, help="output pattern, e.g. runs/sub100_{domain}.trec")
ap.add_argument("--k", type=int, default=100)
a = ap.parse_args()

import glob
import os

paths = sorted(glob.glob(a.runs))
if not paths:
    raise SystemExit("no runs match %s" % a.runs)

for p in paths:
    domain = os.path.basename(p).split("_")[-2]
    rows = collections.defaultdict(list)
    with open(p, encoding="utf-8") as f:
        for ln in f:
            qid, _q0, docid, rank, score, tag = ln.split()
            rows[qid].append((int(rank), score, docid, tag))
    out = a.out.format(domain=domain)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for qid, v in rows.items():
            for rank, score, docid, tag in sorted(v)[:a.k]:
                f.write("%s\tQ0\t%s\t%d\t%s\t%s\n" % (qid, docid, rank, score, tag))
    print("[truncate] %s -> %s (%d topics, k=%d)" % (p, out, len(rows), a.k))
