"""Read the real index statistics (docCount, total term freq, per-term df, norms)
from the prebuilt Lucene index and recompute Lucene's BM25 score for one
document, to check whether the residual gap is explained by index-side statistics.

Lucene 8/9 BM25 (no (k1+1) constant):
    score = sum_t idf(t) * f(t,d) / (f(t,d) + k1 * (1 - b + b * dl(d)/avgdl))
    idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""
import json
import math
import os
import sys

from jnius import autoclass

sys.path.insert(0, r"D:\RETECO\scripts")
from lib_reteco import load_corpus  # noqa: E402

from pyserini import analysis  # noqa: E402
from pyserini.search.lucene import LuceneSearcher  # noqa: E402

INDEX = r"D:\RETECO\indexes\iota\lucene-inverted.reteco-v1.1-track1-iota.20261017.1.6.0"
DOMAIN = r"D:\RETECO\data\track1_tempo\iota"
QID = "2714_798"
DOC = "iota/c5cc10bf_0004.txt"
TARGET = 77.2161
K1, B = 0.9, 0.4

s = LuceneSearcher(INDEX)
reader = s.object.reader
FIELD = "contents"
doc_count = reader.getDocCount(FIELD)
sum_ttf = reader.getSumTotalTermFreq(FIELD)
avgdl_idx = sum_ttf / doc_count
print(f"index docCount={doc_count} sumTotalTermFreq={sum_ttf} avgdl={avgdl_idx:.2f}")

Term = autoclass("org.apache.lucene.index.Term")
analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())
queries = {}
with open(os.path.join(DOMAIN, "examples_dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        queries[r["id"]] = r["query"]
q_toks = analyzer.analyze(queries[QID])

# document text and its indexed length (from norms)
doc_ids, texts = load_corpus(DOMAIN)
d_idx = doc_ids.index(DOC)
raw = s.doc(DOC)
print("stored fields:", list(raw.keys()) if isinstance(raw, dict) else type(raw))
doc_tokens = analyzer.analyze(texts[d_idx])
dl_exact = len(doc_tokens)

# Lucene norms: read the encoded norm for the doc; the decoded value is the
# (quantised) document length
norms = reader.getNormValues(FIELD)
lucene_dl = None
if norms is not None:
    # docids are internal Lucene ids; find the internal id of DOC
    int_id = None
    for i in range(reader.maxDoc()):
        if reader.document(i).get("id") == DOC:
            int_id = i
            break
    if int_id is not None and norms.advanceExact(int_id):
        lucene_dl = norms.longValue()
print(f"doc token count (exact) = {dl_exact}; Lucene encoded norm value = {lucene_dl}")

q_tf = {}
for t in q_toks:
    q_tf[t] = q_tf.get(t, 0) + 1
d_tf = {}
for t in doc_tokens:
    d_tf[t] = d_tf.get(t, 0) + 1

score = 0.0
rows = []
for term, fq in q_tf.items():
    if term not in d_tf:
        continue
    df = reader.docFreq(Term(FIELD, term))
    f = d_tf[term]
    idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
    # Lucene's norm encodes dl/avgdl; approximate with exact dl first
    denom = f + K1 * (1 - B + B * dl_exact / avgdl_idx)
    s_term = idf * f / denom
    score += s_term
    rows.append((term, fq, f, df, round(idf, 3), round(s_term, 3)))

print(f"reconstructed Lucene score (exact dl) = {score:.4f}  vs index {TARGET}")
print("top contributions:", sorted(rows, key=lambda r: -r[-1])[:5])
