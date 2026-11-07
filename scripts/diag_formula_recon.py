"""Reconstruct Lucene's BM25 score from raw token statistics and compare it with
the score the official prebuilt index actually returned.

Lucene 8/9 BM25Similarity:
    idf(t)  = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    score   = sum_t idf(t) * freq(t,d) * (k1 + 1)
                    / (freq(t,d) + k1 * (1 - b + b * dl(d) / avgdl))

gensim LuceneBM25Model (used by the official starter kit):
    idf(t)  = log(N + 1) - log(df(t) + 0.5)
    doc vec = idf(t) * freq / (freq + k1 * (1 - b + b * dl / avgdl))     [no (k1+1)]
    query vec = idf(t) * freq_q / (freq_q + k1 * (1 - b + b * dl_q / avgdl))
    score = dot(doc vec, query vec)   -> idf applied twice, query-length factor
"""
import math
import os
import sys

from gensim.corpora import Dictionary

sys.path.insert(0, r"D:\RETECO\scripts")
from lib_reteco import load_corpus  # noqa: E402

from pyserini import analysis  # noqa: E402

DOMAIN = r"D:\RETECO\data\track1_tempo\iota"
QID = "2714_798"
DOC = "iota/c5cc10bf_0004.txt"
K1, B = 0.9, 0.4
LUCENE_SCORE = 77.2161

queries = {}
with open(os.path.join(DOMAIN, "examples_dev.jsonl"), encoding="utf-8") as f:
    import json
    for line in f:
        r = json.loads(line)
        queries[r["id"]] = r["query"]

doc_ids, texts = load_corpus(DOMAIN)
analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
corpus = [analyzer.analyze(t) for t in texts]
dictionary = Dictionary(corpus)
N = len(corpus)
avgdl = sum(len(d) for d in corpus) / N
dfs = dictionary.dfs

q_toks = analyzer.analyze(queries[QID])
q_tf = {}
for t in q_toks:
    q_tf[t] = q_tf.get(t, 0) + 1
d_idx = doc_ids.index(DOC)
d_toks = corpus[d_idx]
dl = len(d_toks)
d_tf = {}
for t in d_toks:
    d_tf[t] = d_tf.get(t, 0) + 1

lucene_total = 0.0
gensim_total = 0.0
n_terms = 0
for term, fq in q_tf.items():
    tid = dictionary.token2id.get(term)
    if tid is None or term not in d_tf:
        continue
    n_terms += 1
    df = dfs[tid]
    f = d_tf[term]
    idf_l = math.log(1 + (N - df + 0.5) / (df + 0.5))
    idf_g = math.log(N + 1.0) - math.log(df + 0.5)
    lucene_total += idf_l * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / avgdl))
    # gensim: doc weight x query weight (query length = len(q_toks))
    dw = idf_g * f / (f + K1 * (1 - B + B * dl / avgdl))
    qw = idf_g * fq / (fq + K1 * (1 - B + B * len(q_toks) / avgdl))
    gensim_total += dw * qw

print(f"terms in common: {n_terms} | doc length {dl} tokens | avgdl {avgdl:.1f}")
print(f"Lucene formula   : {lucene_total:.4f}   (index returned {LUCENE_SCORE:.4f})")
print(f"gensim formula   : {gensim_total:.4f}   (official pipeline returned 93.5363)")
print(f"ratio lucene/reconstructed-index: {lucene_total / LUCENE_SCORE:.4f}")
