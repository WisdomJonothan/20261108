#!/usr/bin/env python3
"""Verbatim RETECO official BM25 baseline retrieval (starter_kit/official_baseline.py).

The retrieval class below is copied unchanged from
  https://github.com/DataScienceUIBK/RETECO/blob/main/starter_kit/official_baseline.py
(pyserini Lucene analyzer + gensim LuceneBM25Model, k1=0.9, b=0.4, top-1000).
Only the scoring step is external: this script writes a TREC run that the
official scorer (scorer.py / ir_metrics.py) evaluates.

pytrec_eval could not be built on this Windows machine (no MSVC C++ workload,
the sdist downloads and compiles trec_eval at install time). The official
nDCG@10 is therefore computed with the organizers' pure-Python scorer, which is
numerically identical for these binary qrels.

Usage:
  python run_bm25_official.py --data D:/RETECO/data/track1_tempo/iota --split dev \
      --out D:/RETECO/runs/bm25_official_iota_dev.trec
"""
import argparse
import json
import os
import platform
import time

from lib_reteco import load_corpus, load_examples, write_trec_run


class OfficialBM25:  # verbatim from RETECO starter_kit/official_baseline.py
    def __init__(self, documents, doc_ids):
        from pyserini import analysis
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity
        from tqdm import tqdm

        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
        self.doc_ids = doc_ids
        corpus = [self.analyzer.analyze(x) for x in
                  tqdm(documents, desc="  analyze corpus", unit="doc", leave=False)]
        self.dictionary = Dictionary(corpus)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=0.9, b=0.4)
        bm25_corpus = self.model[list(map(self.dictionary.doc2bow, corpus))]
        self.index = SparseMatrixSimilarity(
            bm25_corpus, num_docs=len(corpus), num_terms=len(self.dictionary),
            normalize_queries=False, normalize_documents=False)

    def search(self, queries, query_ids, desc=""):
        all_scores = {}
        for qid, query in zip(query_ids, queries):
            toks = self.analyzer.analyze(query)
            sims = self.index[self.model[self.dictionary.doc2bow(toks)]].tolist()
            pairs = sorted(zip(self.doc_ids, sims), key=lambda x: x[1],
                           reverse=True)[:1000]
            all_scores[str(qid)] = pairs
        return all_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    import importlib.metadata as md

    import gensim

    t0 = time.perf_counter()
    doc_ids, texts = load_corpus(args.data)
    t_load = time.perf_counter()
    bm25 = OfficialBM25(texts, doc_ids)
    t_index = time.perf_counter()
    qids, queries = load_examples(args.data, args.split)
    scores = bm25.search(queries, qids)
    t_search = time.perf_counter()

    ranked = {qid: scores[qid] for qid in qids}
    write_trec_run(args.out, ranked, tag="bm25official")

    meta = {
        "system": "RETECO official BM25 (starter_kit/official_baseline.py, verbatim)",
        "k1": 0.9, "b": 0.4, "topk": 1000,
        "num_docs": len(doc_ids), "num_queries": len(qids),
        "split": args.split, "data": os.path.abspath(args.data),
        "seconds_load": t_load - t0,
        "seconds_index": t_index - t_load,
        "seconds_search": t_search - t_index,
        "gensim_version": gensim.__version__,
        "pyserini_version": md.version("pyserini"),
        "pyjnius_version": md.version("pyjnius"),
        "python": platform.python_version(),
        "run_file": os.path.abspath(args.out),
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[bm25-official] {len(qids)} queries over {len(doc_ids)} docs "
          f"(index {meta['seconds_index']:.1f}s, search {meta['seconds_search']:.1f}s)")


if __name__ == "__main__":
    main()
