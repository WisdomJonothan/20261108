# Partial evidence: what the cross-encoder actually rewards (2026-11-13)

Question asked by the third review: are documents that cover *one step* of a
multi-step question systematically ranked below documents that answer the whole
question? Measured on dev, using the frozen system's cross-encoder, over every
document in that query's dense top-100 (13 domains, 519 queries).

| documents | n | median CE rank | share in CE top-10 |
|---|---|---|---|
| cover all K steps (K>=2) | 157 | **4** | **73.2%** |
| cover 2 steps of K>=3 | 253 | 7 | 59.3% |
| cover 1 step of K>=2 | 462 | **13** | **47.0%** |
| gold for a K=1 query | 85 | 9 | 50.6% |
| not relevant (K>=2) | 40,028 | 51 | 9.0% |

The ranking is graded by coverage: a full answer is almost three times as
likely to be in the top ten as a single-step witness of the same query, and the
median single-step witness sits at rank 13 - just outside the ten slots that
decide nDCG@10. Nothing in the listwise objective asks for coverage, so this is
the behaviour it should have; it is also why 59.5% step coverage is consistent
with a 0.3581 nDCG.

Where a gold document that the pool contains is lost, it is almost always the
cross-encoder, not the fusion:

- 138 dev queries score nDCG@10 = 0. Of these, 86 have a relevant document in
  the dense top-100; 42 more have one only in 101-1000; 10 have none in the
  top-1000 at all.
- 156 (query, gold) pairs have the gold inside the pool the cross-encoder saw
  and outside the fused top ten. In 19 of them the cross-encoder itself put it
  in its own top ten (fusion demoted it); in 96 it ranked 11-50 and in 41
  worse than 50.

So the hypothesis in the review holds on this data: the model sees the
document, and buries the partial witness. The lever is the training target,
not the fusion weight - which is consistent with the fusion ablations already
run (reweighting moves single digits and never past the frozen system).

Files: `results/PARTIAL_EVIDENCE.json` (bucket statistics and the 372 ranked
pairs), `results/HARD_GOLD_PAIRS.json` (the 156 pairs with dense/CE/fused
ranks). A windowing explanation was ruled out: the deployment reranker scores
every window of the document (`run_chunk_rerank.py`, pool max), so evidence is
never outside what the model reads; the `--max-windows 3` cap exists only in
training-time validation.
