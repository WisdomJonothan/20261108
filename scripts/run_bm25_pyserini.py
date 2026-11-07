#!/usr/bin/env python3
"""BM25 retrieval over an official RETECO v1.1 Pyserini/Lucene prebuilt index.

The prebuilt indexes (castorini/prebuilt-indexes-reteco) are Anserini Lucene
indexes of the official deduplicated RETECO v1.1 Track 1 collections. Anserini's
BM25 defaults are k1=0.9, b=0.4, i.e. exactly the constants used by the RETECO
official baseline, so no tuning is performed here.

Usage:
  python run_bm25_pyserini.py --index D:/RETECO/indexes/iota \
      --data D:/RETECO/data/track1_tempo/iota --split dev \
      --out D:/RETECO/runs/bm25_pyserini_iota_dev.trec \
      --meta D:/RETECO/runs/bm25_pyserini_iota_dev.meta.json
"""
import argparse
import json
import os
import platform
import time

from lib_reteco import load_examples, write_trec_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    import importlib.metadata as md

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)

    qids, queries = load_examples(args.data, args.split)
    ranked, latencies, n_hits = {}, [], {}
    for qid, q in zip(qids, queries):
        t0 = time.perf_counter()
        hits = searcher.search(q, k=args.topk)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ranked[qid] = [(h.docid, float(h.score)) for h in hits]
        n_hits[qid] = len(hits)

    write_trec_run(args.out, ranked, tag="bm25pyserini")

    lat = sorted(latencies)
    meta = {
        "system": "Pyserini LuceneSearcher BM25 over official prebuilt index",
        "index_dir": os.path.abspath(args.index),
        "index_num_docs": searcher.num_docs,
        "k1": args.k1,
        "b": args.b,
        "topk": args.topk,
        "split": args.split,
        "data": os.path.abspath(args.data),
        "num_queries": len(qids),
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": lat[len(lat) // 2],
            "p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
            "max": lat[-1],
        },
        "avg_hits": sum(n_hits.values()) / len(n_hits),
        "pyserini_version": md.version("pyserini"),
        "pyjnius_version": md.version("pyjnius"),
        "python": platform.python_version(),
        "run_file": os.path.abspath(args.out),
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[bm25-pyserini] {os.path.basename(args.data)}/{args.split}: "
          f"{len(qids)} queries, {searcher.num_docs} docs, "
          f"mean latency {meta['latency_ms']['mean']:.1f} ms")
    for qid in list(ranked)[:3]:
        print(f"  {qid} top3: {[d for d, _ in ranked[qid][:3]]}")


if __name__ == "__main__":
    main()
