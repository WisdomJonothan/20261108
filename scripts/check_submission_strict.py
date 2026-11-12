#!/usr/bin/env python
"""Strict local sanity check for a TREC submission file.

Usage: check_submission_strict.py <run> <examples.jsonl> [--per-topic N]

Checks that the official format checker does not cover:
  * every query of the query file appears in the run;
  * ranks are 1..n without gaps;
  * scores are strictly decreasing inside a topic (no printed ties);
  * each topic has at least `--per-topic` rows (default 10).
"""
import json
import sys


def main(argv):
    if len(argv) < 3:
        print("usage: check_submission_strict.py <run> <examples.jsonl> [--per-topic N]")
        return 2
    run, ex = argv[1], argv[2]
    need = 10
    if "--per-topic" in argv:
        need = int(argv[argv.index("--per-topic") + 1])
    qids = []
    for line in open(ex, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        qids.append(json.loads(line)["id"])
    rows = {}
    for line in open(run, encoding="utf-8"):
        p = line.split()
        if len(p) < 6:
            continue
        rows.setdefault(p[0], []).append((int(p[3]), float(p[4])))
    missing = [q for q in qids if q not in rows]
    bad = []
    for q in qids:
        r = rows.get(q, [])
        if len(r) < need:
            bad.append((q, "%d rows" % len(r)))
            continue
        if [x[0] for x in r] != list(range(1, len(r) + 1)):
            bad.append((q, "ranks not contiguous"))
        if any(r[i][1] <= r[i + 1][1] for i in range(len(r) - 1)):
            bad.append((q, "scores not strictly decreasing"))
    if missing or bad:
        print(json.dumps({"missing": len(missing), "bad": bad[:5]}, ensure_ascii=False))
        return 1
    print("OK: %d topics x %d docs, contiguous ranks, strictly decreasing scores"
          % (len(qids), min(len(rows[q]) for q in qids)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
