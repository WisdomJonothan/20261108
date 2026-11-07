#!/usr/bin/env python3
"""Warm latency micro-benchmark: is the ~0.4 s/query figure Lucene or the JNI bridge?

Runs each dev query 3 times for k in {10, 100, 1000} after a warmup query, and
reports mean/p50. Separate from the scored runs (which time a cold call).
"""
import argparse
import json
import os
import statistics
import time

from lib_reteco import load_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(0.9, 0.4)
    qids, queries = load_examples(args.data, args.split)

    searcher.search(queries[0], k=10)  # warmup (JIT, class loading)

    out = {"index": os.path.abspath(args.index), "num_queries": len(qids), "k": {}}
    for k in (10, 100, 1000):
        times = []
        for q in queries:
            for _ in range(3):
                t0 = time.perf_counter()
                searcher.search(q, k=k)
                times.append((time.perf_counter() - t0) * 1000.0)
        out["k"][k] = {
            "mean_ms": statistics.mean(times),
            "p50_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["k"], indent=2))


if __name__ == "__main__":
    main()
