"""Official-style BM25 with a switchable query weighting.

The official starter kit transforms the query through the same
`LuceneBM25Model`, so the dot product carries idf twice (query side + doc side).
Lucene applies idf once. This script runs both variants over the same corpus so
the effect can be measured directly.

--query-weight idf    : official behaviour (model applied to the query)
--query-weight unit   : query terms weighted 1.0 (Lucene-like, idf once)
"""
import argparse
import json
import os
import sys

from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_examples, write_trec_run  # noqa: E402

from pyserini import analysis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--query-weight", choices=["idf", "unit"], default="idf")
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
    doc_ids, texts = load_corpus(args.data)
    corpus = [analyzer.analyze(t) for t in texts]
    dictionary = Dictionary(corpus)
    model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
    doc_bows = [dictionary.doc2bow(d) for d in corpus]
    bm25_corpus = model[doc_bows]
    index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus),
                                   num_terms=len(dictionary),
                                   normalize_queries=False,
                                   normalize_documents=False)

    qids, queries = load_examples(args.data, args.split)
    ranked = {}
    for qid, q in zip(qids, queries):
        toks = analyzer.analyze(q)
        bow = dictionary.doc2bow(toks)
        qvec = model[bow] if args.query_weight == "idf" else [(t, 1.0) for t, _ in bow]
        sims = index[qvec]
        pairs = sorted(zip(doc_ids, sims), key=lambda x: -x[1])[: args.topk]
        ranked[qid] = pairs
    write_trec_run(args.out, ranked, tag=f"bm25-{args.query_weight}")
    meta = {"data": os.path.abspath(args.data), "split": args.split,
            "query_weight": args.query_weight, "k1": 0.9, "b": 0.4,
            "num_docs": len(doc_ids), "num_queries": len(qids),
            "run_file": os.path.abspath(args.out)}
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.query_weight}] {len(qids)} queries over {len(doc_ids)} docs")


if __name__ == "__main__":
    main()
