# Independent reviews of the Phase-1 re-ranking pilot

Two sub-agents reviewed the work read-only in this session: one recomputed the
numbers from scratch, one audited the methodology and ran its own diagnostics on
the same candidate pool. Their findings and the action taken are summarised
here; the numbers they quote are reproduced from `runs/` and `results/`.

## A. Independent metric audit

Scope: recompute nDCG@10 from the qrels and the TREC run files with a
from-scratch implementation (two independent implementations, agreeing to
1e-12), verify candidate containment.

Verdict: **the reported numbers are correct to 4 dp.**

| run | recomputed | reported |
| --- | --- | --- |
| dense | 0.217096566591 | 0.2171 |
| rerank@100 | 0.119494395864 | 0.1195 |
| rerank@50 | 0.148150270737 | 0.1482 |

- rerank runs are an exact permutation of the dense top-100 for all 30/30
  queries (|intersection| = 100), so no candidate leakage.
- gold in top-10: dense 23, rerank@100 16, rerank@50 18.
- per-query: rerank better 5 / tie 11 / worse 14.
- caveats raised: 94 gold judgments over 30 queries, 11 queries score 0 for both
  runs, rerank scores are uncalibrated (48% negative) but monotone with rank.

## B. Methodological review

Verdict: implementation is a clean standard cross-encoder pass, **no scoring
bug**, but the original claim ("a generic cross-encoder does not transfer")
overreaches. Its key measurements on the same 3000 pairs:

1. The 512-token window is shared: median available document budget 371 tokens;
   24.7% of pairs cut the document segment, 11.2% cut the query.
2. Truncation is nevertheless **not** the cause: gold docs are short (181 vs 310
   tokens), 96.4% of gold tokens visible, only 8% of gold pairs truncated;
   per-query correlation between truncation rate and loss ≈ 0.
3. No passage-length bias: within-query corr(length, logit) = 0.
4. The right statistic is within-query AUC for gold vs non-gold: dense 0.760,
   bge-reranker-base 0.668 — the re-ranker is a weaker discriminator than the
   retriever it re-ranks.
5. The loss is a depth artefact: re-ranking only the dense top-N gives
   0.2143 (N=5), 0.2064 (10), 0.1692 (20), 0.1482 (50), 0.1195 (100).
6. Statistical strength was missing from the write-up: Δ = −0.0976,
   t = −2.42, bootstrap 95% CI [−0.177, −0.021]; bitcoin train +0.0078 (n.s.).
7. Query form (full forum posts, mean 266 raw tokens) is out-of-distribution for
   MS-MARCO re-rankers and was never varied — recommended as the first test.

Recommended replacement claim (adopted): *"Under a 512-token shared budget, full
replacement of the dense top-100 with bge-reranker-base lowers Bitcoin-dev
nDCG@10 from 0.2171 to 0.1195 (95% CI [−0.177, −0.021]); damage scales with
re-rank depth."*

## C. What was done in response

| review point | action | result |
| --- | --- | --- |
| shared-window truncation | new `run_chunk_rerank.py` with explicit query cap + document windows + max-pool | chunking does not help (0.1067 vs 0.1089), matching the review's prediction |
| depth artefact | depth ladder reproduced; claim re-scoped to full replacement | note §4.5 |
| query form untested | ran title / guidance-summary / clipped variants for three models | title is best for all three; full-post queries are indeed the weak spot |
| one checkpoint only | added MiniLM-L6 and bge-reranker-v2-m3 | v2-m3 title 0.2121 ≈ dense 0.2171 (n.s.); MiniLM 0.2041 (n.s.) |
| significance | paired bootstrap on every comparison, dev and train | see note §2–3 |
| tokenizer bug (found independently) | MiniLM run redone after the complete vocab.txt arrived | 0.0043 → 0.1704 (the 0.0043 was an artefact and is discarded) |
