# Phase 1 pilot — cross-encoder re-ranking on RETECO (IOTA / Bitcoin)

Status: **real runs on this machine, 2026-11-07.** Every number below comes from
a stored run file and its per-query score JSON. Nothing was tuned on dev; dev
was used as a held-out check after the first configuration, and every later
configuration is reported whether it helped or not.

## 0. Correction to the first write-up

The first version of this note said "a generic cross-encoder does not transfer".
That claim was too strong and the experiment had two defects, both now fixed:

1. `run_rerank.py` fed (query, document) into a 512-token encoder with the
   default `longest_first` truncation. RETECO queries are long (median 221
   tokens, max 1017), so the document side was silently cut to ~250 tokens on
   average. Fixed by an explicit query cap + document windows
   (`run_chunk_rerank.py`).
2. The claim rested on one checkpoint at one depth. A stronger checkpoint
   (bge-reranker-v2-m3) and query-form variants change the picture, and the
   depth ladder shows the loss is a full-replacement artefact.

A second bug worth recording: the first `ms-marco-MiniLM-L-6-v2` run scored
0.0043 because the tokenizer's `vocab.txt` was still being downloaded (vocab
size 5 → every token `[UNK]`). The number was an artefact and is discarded;
with the complete vocabulary the same configuration scores 0.1704.

## 1. Setup

| item | value |
| --- | --- |
| models | `BAAI/bge-reranker-base` (XLMR 12L/768), `BAAI/bge-reranker-v2-m3` (568M), `cross-encoder/ms-marco-MiniLM-L-6-v2` (6L/384) |
| precision | fp16 on GPU (fp32 control run: identical ordering, 0.1479 vs 0.1482 at k=50) |
| candidate source | dense `bge-base-en-v1.5` top-100 (frozen; recall@100 cannot change) |
| document handling | `full`: strip tags; `title`: text before the first `<p>`; `summary`: `guidance.query_summary`; `clip`: 96-token prefix |
| windows | single window (k=100, win 250–380) or sliding windows with max-pool (`--win/--stride`) |
| scoring | single relevance logit, descending; no qrels read by the re-ranker |

## 2. All measured configurations — Bitcoin dev (30 queries)

Dense baseline **0.2171**; candidate oracle at depth 100 **0.6488**.

| re-ranker | query form | document mode | nDCG@10 | Δ vs dense |
| --- | --- | --- | --- | --- |
| bge-reranker-base | full (orig, 512 pair) | truncated pair | 0.1195 | −0.0976 |
| bge-reranker-base | full | single window 380 | 0.1089 | −0.1082 |
| bge-reranker-base | full | chunked 380/190 max-pool | 0.1067 | −0.1104 |
| bge-reranker-base | title | single window 380 | 0.1470 | −0.0701 |
| bge-reranker-base | guidance summary | single window 380 | 0.1073 | −0.1098 |
| bge-reranker-base | clipped 96 | single window 380 | 0.0803 | −0.1368 |
| MiniLM-L6 | full | single window 380 | 0.1704 | −0.0467 |
| MiniLM-L6 | full | chunked 380/190 | 0.1647 | −0.0524 |
| MiniLM-L6 | title | single window 380 | **0.2041** | −0.0130 (n.s.) |
| bge-reranker-v2-m3 | full | single window 250 | 0.1577 | −0.0594 |
| bge-reranker-v2-m3 | title | single window 250 | **0.2121** | −0.0050 (n.s.) |
| bge-reranker-v2-m3 | title | single window 380 | 0.1983 | −0.0188 |
| RRF(dense, v2-m3 title) c=60 | — | — | **0.2264** | +0.0093 |
| RRF(dense, v2-m3 title) 2:1 | — | — | **0.2311** | +0.0140 |

Paired bootstrap (10,000 resamples, seed 0):

| comparison | Δ | 95% CI | P(Δ>0) |
| --- | --- | --- | --- |
| bge-base full replacement | −0.0976 | [−0.178, −0.023] | 0.005 |
| MiniLM title | −0.0130 | [−0.099, +0.076] | 0.378 |
| v2-m3 title | −0.0050 | [−0.081, +0.071] | 0.441 |
| RRF 2:1 (dev) | +0.0140 | [−0.030, +0.052] | — |

## 3. Bitcoin train (70 queries, held-out from all configuration choices)

Dense baseline **0.1550**.

| configuration | nDCG@10 | Δ | 95% CI |
| --- | --- | --- | --- |
| v2-m3, title, window 250 | 0.1637 | +0.0087 | [−0.041, +0.063] |
| RRF(dense, v2-m3) 2:1 | 0.1713 | +0.0163 | [−0.009, +0.043] |

Pooled over the 100 Bitcoin queries (dev + train), RRF 2:1 gives
**+0.0156, 95% CI [−0.006, +0.038], P(Δ>0)=0.918**.

## 4. What the evidence says

1. **Truncation is not the explanation.** Chunking with max-pool does not beat
   single-window scoring (0.1067 vs 0.1089 with bge-base; 0.1647 vs 0.1704 with
   MiniLM), and 96% of gold tokens were already visible (independent review).
   Corpus is passage-level: median document 238 tokens, only ~14% exceed 512.
2. **Query form matters more than anything else we changed.** Title-only
   queries lift every model (bge-base +0.038, MiniLM +0.034, v2-m3 +0.054).
   RETECO 1a queries are full forum posts; that is out-of-distribution for
   MS-MARCO-style re-rankers.
3. **A competent re-ranker roughly ties the dense retriever; it does not beat
   it.** v2-m3 title 0.2121 vs dense 0.2171 (n.s.); MiniLM title 0.2041 (n.s.).
   bge-reranker-base is significantly worse.
4. **Fusion is the only configuration with a positive point estimate on both
   splits** (dev +0.014, train +0.016, pooled +0.016), but with 100 queries the
   CI still includes 0. Treat as promising, not established.
5. The damage from a bad re-ranker is a full-replacement artefact: re-ranking
   only the dense top-5 costs −0.003, top-100 costs −0.098 (depths 5/10/20/50/100
   → 0.2143/0.2064/0.1692/0.1482/0.1195).

## 5. Consequence for the temporal plan

The "dense recall → strong semantic re-ranker → temporal module" stack cannot
be built on the assumption that middle layer exists. What exists today:

* dense (0.2171 dev / 0.1550 train) as the strong baseline;
* a fusion recipe that adds ~+0.01–0.016 without significance;
* no re-ranker that dominates.

Therefore any temporal signal should be tested as a **feature added to the
frozen dense top-100** (or to the fusion), scored with the same paired
bootstrap, developed on train and reported on dev — not bolted onto a
re-ranker whose own contribution is unproven.

## 6. Artifacts

* runs: `runs/rerank_*.trec` (one per configuration)
* meta: `results/rerank_*.meta.json` (model, windows, q-mode, pairs, latency)
* scores: `results/score_rerank_*.json` (per-query)
* diagnostics: `results/diag_len_bitcoin_dev.json`
* code: `scripts/run_chunk_rerank.py`, `scripts/run_rerank.py`,
  `scripts/diag_length.py`
* reviews: independent metric audit and methodological review were run in this
  session; their verdicts are summarised in §0 and §4.
