# RETECO Track 1a — system paper draft

Status: internal draft, 2026-11-08. All numbers are dev-split results produced by
the committed scripts; the hidden test set has not been used. Sections marked
[OPEN] need a decision or extra experiment before submission.

---

## Abstract (draft)

Temporal retrieval asks for documents that are not only topical but consistent
with the time conditions of the query. We describe our system for the RETECO
Track 1a shared task: a two-view pipeline that combines a strong dense retriever
(`mxbai-embed-large-v1`) with a cross-encoder reranker (`bge-reranker-v2-m3`) by
reciprocal rank fusion, with the fusion weight selected on the training split. On
the 13 development domains (519 queries) the system reaches a domain-macro
nDCG@10 of 0.3176, against 0.2943 for dense-only and 0.1147 for the official BM25
baseline. We further report a fully pre-registered study of temporal modelling on
this benchmark: a temporal-consistency cross-encoder score, a train-fitted gate,
query decomposition and HyDE all fail to improve over the plain cross-encoder,
and a year-consistency sensitivity analysis explains why — on RETECO, a document
whose year does not match the query is very often still relevant.

## 1 Introduction

RETECO (Reasoning-Oriented Retrieval) Track 1a provides 13 domain corpora with
queries whose gold documents must satisfy both a topic condition and a temporal
condition expressed in the query. Three time notions coexist — query time,
publication time and content time — and the corpus ships no timestamps, so
temporal reasoning has to happen on text.

We entered the track with two questions: (i) how much retrieval quality can be
obtained without task-specific training, and (ii) whether explicit temporal
modelling adds anything on top of a strong cross-encoder. The paper answers the
first positively and the second negatively, with a pre-registered protocol.

## 2 Task and data

* v1.1 release: 13 domains, 1,167,159 documents in total (verified locally:
  the per-domain document counts and the 519 dev queries sum exactly to the
  official totals).
* Split: 1,211 train / 519 dev queries; per domain between 3 (IOTA) and 240
  (history) dev queries.
* Metrics: official `ir_metrics.py` — nDCG@10 (binary relevance), plus temporal
  precision/recall/coverage and an aggregate NDCG|FC. Relevance labels are binary.
* Retrieval is per domain: a query only searches its own corpus. We never claim
  "1.17 M documents per query".

## 3 System

### 3.1 First stage

`mxbai-embed-large-v1` (335M, 1024-d), mean pooling, query instruction
"Represent this sentence for searching relevant passages: ", fp16, 512 tokens,
top-1000. Documents are encoded once per domain and cached.

Choice of encoder was empirical: bge-base-en-v1.5 (0.2486 macro), e5-base-v2
(0.1480), mxbai (0.2942), all zero-shot on the same split. gte-large-en-v1.5
could not be loaded under transformers 5.22 (custom RoPE/pack_qkv).

### 3.2 Second stage

`bge-reranker-v2-m3` scores the dense top-100. The query view for the reranker is
the title-like prefix of the query (up to the first `<p>`), capped at 256 tokens,
and long documents are split into 250-token windows with max-pooling over
windows. This configuration was fixed on bge-base and reused unchanged for mxbai.

### 3.3 Fusion

Reciprocal rank fusion, `score(d) = w_d/(60+rank_dense) + w_ce/(60+rank_ce)`.
The weight is chosen **on train only** (13-domain train macro picks dense:CE =
1:2); dev is read once. Per-domain weights do not transfer — choosing a weight on
each domain's own train split collapses the dev macro to 0.2941, i.e. dense level.

## 4 Results (dev nDCG@10)

| system | macro | notes |
|---|---:|---|
| official BM25 (published / reproduced) | 0.1147 | official starter kit, exact reproduction |
| Anserini BM25 on prebuilt index | 0.0967 | same data, Lucene scoring |
| bge-base dense | 0.2486 | |
| mxbai dense | 0.2942 | |
| mxbai + CE, RRF 2:1 | **0.3176** | weight chosen on dev (optimistic) |
| mxbai + CE, RRF 1:2 | 0.3134 | weight chosen on train (honest) |

Paired bootstrap (13 domains, 519 queries) for RRF 2:1 vs dense: +0.0271
[+0.0163, +0.0377]; for the train-selected 1:2: +0.0334 [+0.0164, +0.0503]
query-level, but the domain-clustered CI [−0.0162, +0.0541] crosses zero. We
report the point estimate and state the limitation explicitly.

## 5 Temporal analysis (all negative, pre-registered where noted)

1. **Observability.** Gold documents contain a year in 84.7 % (politics) … 9.3 %
   (cardano) of cases; query-side year anchors cover 6.7–57.8 %. Earlier temporal
   modules were therefore evaluated mostly where temporal consistency is almost
   unobservable.
2. **Year-consistency sensitivity.** Reordering the dense top-100 by
   query-year/document-year consistency without qrels *decreases* nDCG@10 in 6 of
   8 tested domains (politics 0.307→0.268, hsm 0.293→0.271, bitcoin 0.217→0.197)
   and helps only economics (0.189→0.201). Among year-bearing candidates 53–100 %
   carry a year outside the query interval, and many are gold. **Time mismatch is
   not irrelevance.**
3. **Discriminator.** A cross-encoder score against a temporal statement built
   from guidance fields is discriminative (AUC 0.61–0.88 vs 0.5 placebo).
4. **Gate (pre-registered).** Fitting a logistic gate on train (1,857
   topic-hard/time-hard pairs), choosing λ inside train, and reading dev once gives
   ΔnDCG@10 +0.0039 [−0.0023, +0.0103] — far below the +0.02 adoption bar.
5. **Other temporal attempts.** Query decomposition (oracle steps and Qwen2.5-1.5B),
   HyDE, content-time features and a semantic temporal-consistency score are all
   neutral or negative.

Claim: text-only temporal signals exist in this benchmark but do not convert into
ranking gains under a separated protocol; the benchmark's own relevance labels are
weakly aligned with year consistency.

## 6 Discussion and limitations

* Dev only; the hidden test queries are unseen. All weights are chosen on train,
  which is why we report both the dev-optimal (0.3176) and train-selected (0.3134)
  configurations.
* Small domains (IOTA 3, quant 10, law 11, workplace 11 dev queries) make per-domain
  numbers noisy; macro averages are dominated by history (240 queries).
* Domain-clustered variance is large; the fusion gain is not established beyond dev.
* No 4B–7B retrievers were run (8 GB GPU); TEMPO's 30.4/32.0 numbers are on a
  different 1,730-query v1.0 corpus and are not directly comparable.
* Runtime: dense encoding 46–150 doc/s depending on domain; CE reranking ~2 s/query.

## 7 Reproducibility

Public repository: https://github.com/WisdomJonothan/20261108 — scripts, TREC runs
for every system, per-domain score JSONs, reports. Paths are resolved from
`RETEco_ROOT`/`RETEco_MODELS`; scoring uses the vendored official `ir_metrics.py`.

[OPEN] A paragraph on the submitted test run (once the hidden queries are released)
and the camera-ready numbers.

## References (to complete)

BGE, mxbai-embed, Anserini/Lucene BM25, RRF (Cormack et al. 2009), TEMPO
(arXiv:2601.09523v1), TEMPO/RETECO overview paper [OPEN].
