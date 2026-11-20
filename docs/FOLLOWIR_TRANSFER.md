# FollowIR transfer: plan and status (2026-11-20, revised)

## What FollowIR is

FollowIR (Weller et al., 2024) measures whether a reranker follows *instructions*
attached to a TREC-style topic. Each topic carries an original description, a
modified one, and two qrels files; a system is evaluated with the original and
with the modified instruction, and the reported quantity is the *pairwise*
difference (the model should improve on the qrels its instruction matches and
not on the other). The three official collections are **Robust04, Core17 and
News21** - not MSMARCO. Each official data release ships the corpus, the topics,
both qrels and the official candidate lists, so no external collection is
needed.

An earlier version of this document claimed the evaluation data required tens
of GB of MSMARCO text and was therefore out of reach on this host. That was
wrong: the official repository is `orionw/FollowIR` (branch `master`), and the
datasets are on Hugging Face as

| release | size | contents |
|---|---:|---|
| `jhu-clsp/core17-instructions` | 47.1 MB | corpus, 20 topics (og + changed), two qrels, top-1000 candidates |
| `jhu-clsp/news21-instructions` | 81.7 MB | same shape |
| `jhu-clsp/robust04-instructions` | 118 MB | same shape |

Total ≈ 247 MB, downloaded through the workstation proxy (the compute host has
no direct egress). Core17 is already local at `D:\RETECO\followir\core17`
(45 MB): `corpus.jsonl` (19,899 documents), `queries.jsonl` (20 topics with
`text`, `instruction_og`, `instruction_changed`, `short_query`),
`qrels_og/test.tsv`, `qrels_changed/test.tsv` (4,741 judgements each) and
`top_ranked.jsonl` (20 × 1,000 BM25 candidates).

## Protocol we will run (frozen before looking at scores)

1. Rank the 19,899 Core17 documents for each of the 20 topics with the frozen
   dense retriever (mxbai-embed-large-v1, same query prefix and pooling as the
   RETECO runs), keeping the top 100.
2. Re-rank those 100 with the frozen cross-encoder checkpoint
   (`results/ce_ft/v2m3_pool8_e2/best`, window 250 / stride 192 / at most 3
   windows, pair length 510), z-normalise both score vectors and fuse 1:1 -
   the exact deployment configuration of the RETECO system.
3. Evaluate four arms with the same candidate generator and decision rule as
   the dev study (cap 3, ≤9 pairs, both orders, threshold 0, one swap):
   frozen, Direct, cheap-gap pair selection, and the corrected Qwen3-Reranker
   (single swap and full rerank).
4. Report nDCG@10 under the original instruction and under the changed
   instruction, per collection, plus the paired original-minus-changed
   difference that FollowIR uses; also report the 13-domain RETECO breakdown as
   a within-domain sanity check.

The strategy parameters (candidate cap, pair budget, threshold) are transported
unchanged from the frozen RETECO protocol; nothing is tuned on FollowIR.

## Overlap control

FollowIR topics are Robust04/Core17/News21 TREC topics, not RETECO queries, so
there is no RETECO-train/dev query overlap. The cross-encoder was trained on
RETECO corpus text only, so a FollowIR number is a zero-shot transfer number
for the reranker as well. Document overlap between Core17 and the RETECO
corpora is not possible (different collections), so no de-duplication step is
needed.

## Status

- Core17 data: downloaded and validated (20 topics, 19,899 documents, two qrels
  files, 20,000 candidate pairs).
- News21 and Robust04: not downloaded yet (81.7 MB and 118 MB).
- Evaluation harness: to be adapted from `scripts/run_chunk_rerank_qvar.py` +
  `scripts/fusion_for_train.py`; the reranker judge is reused unchanged.
- This transfer run is the last outstanding item of the six-step program; it is
  reported separately from the dev table and does not feed back into any dev
  choice.
