"""Why does the prebuilt Lucene index not reproduce the official gensim baseline?

Hypothesis under test: gensim's LuceneBM25Model is applied to the *query* as well
as to the documents (`BM25ABC.__getitem__` is called for both), so the dot
product between the two vectors contains idf TWICE, while Lucene's BM25 applies
idf once. Query-side idf is constant per query for a given term but varies
*across terms*, so it re-weights terms inside the query and changes the ranking.

Test: for one query, compute the score of the same document under
  (a) the official pipeline:      q_vec = model[doc2bow(query)]   (idf included)
  (b) idf-free query vector:      q_vec = [(term_id, 1.0), ...]   (idf once)
and compare both with the score the Lucene index actually produced.
"""
import json
import os
import sys

from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity

sys.path.insert(0, r"D:\RETECO\scripts")
from lib_reteco import load_corpus  # noqa: E402

from pyserini import analysis  # noqa: E402

DOMAIN = r"D:\RETECO\data\track1_tempo\iota"
QID = "2714_798"
DOC = "iota/c5cc10bf_0004.txt"

doc_ids, texts = load_corpus(DOMAIN)
analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
corpus = [analyzer.analyze(t) for t in texts]
dictionary = Dictionary(corpus)
model = LuceneBM25Model(dictionary=dictionary, k1=0.9, b=0.4)
doc_bows = [dictionary.doc2bow(d) for d in corpus]
bm25_corpus = model[doc_bows]  # doc-side weights (idf included)
index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus),
                               num_terms=len(dictionary),
                               normalize_queries=False,
                               normalize_documents=False)

queries = {}
with open(os.path.join(DOMAIN, "examples_dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        queries[r["id"]] = r["query"]
toks = analyzer.analyze(queries[QID])
qbow = dictionary.doc2bow(toks)

q_official = model[qbow]                       # idf included (official pipeline)
q_unit = [(tid, 1.0) for tid, _ in qbow]       # idf once (Lucene-like)

di = doc_ids.index(DOC)
sims_official = index[q_official]
sims_unit = index[q_unit]
print(f"query {QID}: {len(qbow)} distinct terms, {len(toks)} tokens")
print(f"official-style score for {DOC}: {sims_official[di]:.4f}")
print(f"idf-free  query score for {DOC}: {sims_unit[di]:.4f}")
print(f"Lucene index score for {DOC}:  77.2161  (recorded in runs/bm25_pyserini_iota_dev.trec)")

# idf re-weighting inside the query: how much do term weights vary?
import numpy as np
weights = np.array([w for _, w in q_official])
print(f"query term weights: min {weights.min():.3f} max {weights.max():.3f} "
      f"ratio {weights.max()/max(weights.min(),1e-9):.1f}")

# rank correlation between the two query weightings over the whole corpus
ra = np.argsort(-sims_official)[:100]
rb = np.argsort(-sims_unit)[:100]
print(f"top-100 overlap between idf-query and unit-query: {len(set(ra) & set(rb))/100:.2f}")
