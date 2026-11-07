# Phase 2 — cross-domain check (Economics) and LLM re-ranking

Status: real runs on this machine, 2026-11-07. All numbers come from `runs/` and
`results/`; official published values are marked. Corrected after an independent
review; see §6 for the change list.

## 1. Third domain: Economics

Why Economics: 74,692 documents (v1.1), 83 queries (58 train / 25 dev), same
StackExchange provenance and record format as the crypto domains, and a
published BM25 dev nDCG@10 of 0.0467.

Pipeline:
* corpus: `data/track1_tempo/economics/documents.jsonl` (74,692 docs);
* BM25: locally built Anserini Lucene index (`indexes/econ_lucene`), k1=0.9,
  b=0.4, top-1000;
* dense: `BAAI/bge-base-en-v1.5`, same encoder settings as the other domains;
* pointwise re-rank: `bge-reranker-v2-m3`, title-form query, window 250;
* listwise: `Qwen2.5-1.5B-Instruct`, window 20, stride 10;
* pairwise: same model, round-robin over the dense top-10.

| system | dev (25 q) | train (58 q) |
| --- | --- | --- |
| BM25 (local Anserini Lucene) | 0.0269 | 0.0361 |
| BGE dense (zero-shot) | **0.1891** | **0.1209** |
| pointwise v2-m3 (title query, top-100) | 0.1664 | — (not run) |
| candidate oracle at depth 100 | 0.6460 | 0.6008 |
| published BM25 dev (official gensim implementation) | 0.0467 | — |

Deltas, per domain (do not mix them):
* Economics: dense − BM25 = **+0.162 dev / +0.085 train**.
* Bitcoin (Phase 0, official-reproduced BM25): dense − BM25 = **+0.180 dev /
  +0.069 train** (0.2171 vs 0.0370; 0.1550 vs 0.0856).

Two honesty notes:

1. **The local BM25 is not the official baseline.** Economics local BM25 dev is
   0.0269 against the published 0.0467 (−42%), the same kind of gap already seen
   on Bitcoin (local 0.0300 vs published 0.0370). The cause is the one documented
   in Phase 0: Anserini Lucene BM25 vs the organizers' gensim `LuceneBM25Model` —
   the analyzer matches, the scoring implementation does not. We never claim to
   reproduce the published BM25 for Economics.
2. Candidate-oracle ratios: Bitcoin 0.6488 / 0.2171 = **3.0×**, Economics
   0.6460 / 0.1891 = **3.4×** — the same shape, but each ratio belongs to its
   own domain.

What replicates:
* dense ≫ BM25 in both domains (deltas above);
* large ranking headroom in both (3.0× / 3.4× to the depth-100 oracle);
* pointwise re-ranker does not beat dense: Economics −0.023 (0.1664 vs 0.1891),
  Bitcoin −0.005 n.s. (0.2121 vs 0.2171). *(There is no IOTA v2-m3 pointwise
  run; IOTA's earlier −0.088 came from bge-reranker-base, a different
  checkpoint.)*

## 2. Listwise LLM re-ranking: what actually happens

Model: `Qwen2.5-1.5B-Instruct`, fp16, greedy decoding, RankGPT-style prompt
(numbered passages + "rank them most relevant first"), window 20, stride 10.

| run | nDCG@10 | dense baseline |
| --- | --- | --- |
| IOTA dev, top-20 listwise | 0.4712 | 0.4712 |
| Bitcoin dev, top-20 listwise | 0.2171 | 0.2171 |

Diagnostics (this is the evidence, not the identical scores):
* the raw generation is literally `[1] > [2] > ... > [20]`;
* parse failures: **0/30 windows** (`results/listwise_qwen_bitcoin_dev.meta.json`),
  so the identical scores are not a parser-fallback artefact;
* **reversed-input control**: presenting the same 20 passages in reversed order
  still yields `[1] > [2] > ... > [20]` — the model reports input *positions*,
  not a relevance order.

Scope: with this checkpoint and this prompt, listwise re-ranking is a no-op. We
have not shown that listwise re-ranking cannot work — a larger model or a
different prompt may behave differently. What we can say is that a 1.5B instruct
model cannot be used this way.

A direct pairwise probe of the same model (60 pairs = 30 queries × 2 orders,
gold vs random non-gold from the dense top-100) gives:

| metric | value |
| --- | --- |
| pairwise accuracy | **0.717** |
| always-pick-A rate | 0.517 |
| accuracy with gold first | 0.733 |
| accuracy with non-gold first | 0.700 |

## 3. Pairwise LLM re-ranking (round-robin over dense top-10)

Protocol: all 45 pairs per query, deterministic order randomisation, win count,
ties broken by the dense order. Candidate set = dense top-10 — identical for the
baseline, so nDCG@10 is directly comparable and Recall@10 is unchanged.

| domain / split | dense top-10 | pairwise LLM | Δ | bootstrap |
| --- | --- | --- | --- | --- |
| Bitcoin dev (30 q, 1350 calls) | 0.2171 | **0.2602** | +0.0431 | t=+1.22, 95% CI [−0.021, +0.116], P(Δ>0)=0.89 |
| Economics dev (25 q, 1125 calls) | 0.1891 | **0.2082** | +0.0191 | t=+1.13, 95% CI [−0.012, +0.055], P(Δ>0)=0.87 |
| pooled (55 q) | — | — | **+0.0322** | t=+1.54, 95% CI [−0.007, +0.077], P(Δ>0)=0.945 |

Per-query: Bitcoin better 8 / worse 7 / tie 15; Economics better 9 / worse 4 /
tie 12. Positive on both domains, **not significant** at 55 queries.

Budget note: this is a depth-10 experiment. The pointwise numbers in §1 are at
depth 100; the dense baseline is the same run, but the candidate budget differs
(depth-10 re-ranking can only reorder the first ten), so the two must not be
read as one comparison.

Cost: Bitcoin 353 s for 1,350 calls = **11.8 s/query** (Economics 11.2 s/query),
i.e. ~2,600× the dense search latency (4.5 ms).

Positional agreement with the dense order is 12.7% — near the ~10% chance level,
so by itself it is weak evidence of meaningful reordering; read it together with
the nDCG point estimate and the 15/30 exact ties.

## 4. What Phase 2 establishes

1. **Cross-domain replication of the core picture**: dense ≫ BM25, large ranking
   headroom, and a pointwise re-ranker that does not beat dense — all three hold
   on Economics as on the crypto domains. (BM25 levels differ from the published
   ones for the implementation reason in §1.)
2. **Small listwise LLMs are unusable in this configuration**: the 1.5B model
   echoes the input order (0 parse failures; reversed-input control confirms).
3. **Pairwise LLM re-ranking has the largest positive point estimate** of any
   method tried (+0.043 Bitcoin, +0.019 Economics, pooled +0.032, P(Δ>0)=0.945),
   but it is not statistically significant and costs seconds per query. It is
   not the "first method that beats dense": RRF(dense, v2-m3) at 0.2311 also
   beats dense 0.2171 on Bitcoin dev, with a smaller gain.
4. Consequence for the temporal plan: a strong semantic re-rank layer is
   plausible only with a pairwise (or much larger listwise) model, and its cost
   is seconds per query rather than milliseconds.

## 4b. Verification checks performed

* Economics dense dev, official `scorer.py`: nDCG@10 = 0.1891 — identical to
  `score_run.py` (0.1891).
* Pairwise run candidate set equals the dense top-10 for **30/30** queries
  (set equality); listwise order equals the dense top-20 order for **30/30**.
* All rerank/listwise/pairwise runs are restricted to their input candidate
  sets (checked programmatically).

## 5. Files

* `runs/pairwise_qwen_{bitcoin,economics}_dev_k10.trec`,
  `results/pairwise_qwen_*_k10.meta.json`
* `runs/listwise_qwen_{iota,bitcoin}_dev.trec`, `results/listwise_qwen_*.meta.json`
* `results/probe_pairwise_qwen_bitcoin_dev.json`
* `runs/dense_bge_economics_{dev,train}.trec`, `runs/bm25_pyserini_economics_*.trec`
* `runs/rerank_v2m3_title_economics_dev.trec`
* `indexes/econ_lucene` (Anserini, 74,692 docs)

## 6. Corrections made after independent review

| issue raised | fix |
| --- | --- |
| Bitcoin deltas were mislabelled as Economics' (+0.162/+0.085) | §1 lists each domain's own deltas |
| "consistent with … IOTA" unsupported (no IOTA v2-m3 run) | removed; IOTA's −0.088 attributed to bge-reranker-base |
| "same shape as Bitcoin (3.4×)" used Economics numbers | §1 states Bitcoin 3.0× and Economics 3.4× separately |
| BM25 gap to published never reconciled | §1 note 1 explains the implementation difference |
| pairwise (depth 10) vs pointwise (depth 100) mixed | §3 budget note added |
| "first method that beats dense" | corrected: RRF also beats dense; pairwise is the largest gain |
| no-op claim vs general claim | §2 scoped to checkpoint+prompt; added parse-failure and reversed-input evidence |
| "hundreds of queries per second" unsupported | removed |
| 11.0 s/query arithmetic | corrected to 11.8 s/query and ~2,600× |
| 12.7% agreement used as strong evidence | downgraded and read with the 15/30 ties |
