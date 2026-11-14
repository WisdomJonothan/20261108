#!/usr/bin/env python3
"""Splice rows for given qids from one TREC run into another (in place).

Used for the few dev queries whose question alone comes close to the Qwen
input budget: they are re-scored with a larger `--max-len` and their rows
replace the truncated ones in the 13-domain pass.
"""
import argparse
import sys


def read(path):
    rows = []
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) >= 6:
            rows.append((p[0], int(p[3]), p[2], float(p[4])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", required=True)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--qids", required=True)
    a = ap.parse_args()
    keep = {x.strip() for x in a.qids.split(",") if x.strip()}
    main_rows = read(a.into)
    src_rows = read(a.src)
    got = {r[0] for r in src_rows}
    missing = keep - got
    if missing:
        sys.exit("source run lacks qids: %s" % sorted(missing))
    merged = [r for r in main_rows if r[0] not in keep] + [r for r in src_rows if r[0] in keep]
    merged.sort(key=lambda r: (r[0], r[1]))
    with open(a.into, "w", encoding="utf-8") as fh:
        for qid, rank, doc, score in merged:
            fh.write("%s Q0 %s %d %.6f qwen3\n" % (qid, doc, rank, score))
    print("merged %d rows for %d qids into %s" % (len(src_rows), len(keep), a.into))


if __name__ == "__main__":
    main()
