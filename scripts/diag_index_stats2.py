"""Reconstruct the prebuilt index's BM25 score from its own statistics.

Uses pyserini's LuceneIndexReader (getDocCount / getSumTotalTermFreq /
get_term_counts / get_document_vector) so both the idf and the document length
come from the index, not from the gensim analyzer.
"""
import math
import sys

sys.path.insert(0, r"D:\RETECO\scripts")

from pyserini.search.lucene import LuceneSearcher  # noqa: E402

INDEX = r"D:\RETECO\indexes\iota\lucene-inverted.reteco-v1.1-track1-iota.20261017.1.6.0"
DOC = "iota/c5cc10bf_0004.txt"
QUERY = "How long do Zero Value Transactions last?"
TARGET = 77.216103
K1, B = 0.9, 0.4

s = LuceneSearcher(INDEX)
ir = s.index_reader
FIELD = "contents"
N = ir.reader.getDocCount(FIELD)
sum_ttf = ir.reader.getSumTotalTermFreq(FIELD)
avgdl = sum_ttf / N
vec = ir.get_document_vector(DOC)
dl = sum(vec.values())
print(f"N={N} sumTTF={sum_ttf} avgdl={avgdl:.2f} doc_dl={dl} distinct={len(vec)}")

# analyze the query with the index's analyzer
terms = ir.analyze(QUERY)
print("query terms:", terms)

score = 0.0
for t in terms:
    df, cf = ir.get_term_counts(t)
    f = vec.get(t, 0)
    if not df or not f:
        continue
    idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
    st = idf * f / (f + K1 * (1 - B + B * dl / avgdl))
    score += st
    print(f"  {t:15s} df={df:6d} tf={f} idf={idf:.3f} score={st:.3f}")
print(f"reconstructed = {score:.4f}; searcher returned {TARGET}")

# and what Anserini itself computes for this pair
print("compute_query_document_score =", ir.compute_query_document_score(DOC, QUERY))
