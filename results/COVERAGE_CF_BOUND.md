# Counterfactual coverage bound: what a coverage-aware reranker could get for free

Date: 2026-11-14, corrected 2026-11-14 (exact optimisation; see "Correction").
Script: `scripts/coverage_cf_bound.py`.
Outputs: `results/COVERAGE_CF_BOUND.json` (fused top ten),
`results/COVERAGE_CF_BOUND_ce.json` (cross-encoder's own top ten).

The third review asked for the counterfactual bound before committing to a
coverage-aware training objective: keep the relevance pattern of the top ten
fixed (the same judged-relevant documents at the same ranks, so nDCG@10 cannot
change), and only re-choose *which* relevant documents fill those slots, from
the pool the cross-encoder actually sees (dense top-100).

Numbers are per-query averages of annotated steps covered in the top ten,
domain-macro over the 13 dev domains, 519 queries.

| ranking scored | steps in top ten | best with the same slots | pool ceiling | annotation ceiling | nDCG@10 |
|---|---|---|---|---|---|
| fused system | **1.2702** | **1.3405** | 1.6967 | 2.0739 | 0.3581 |
| cross-encoder alone | **1.1122** | **1.1844** | 1.6967 | 2.0739 | 0.2823 |

"best with the same slots" = for each query, hold the number of relevant
documents in the top ten fixed, and pick the most step-complementary set of
that size from the pool: 47 of 519 queries gain (56 step-slots) for the fused
list, 51 queries (67 slots) for the cross-encoder. That is **+0.070 and +0.072
steps per query, +5.5% and +6.5% relative** — the largest coverage gain
available without touching relevance.

## Correction (2026-11-14)

The first version reported 1.2989 / 1.1593 (+2.3% / +4.2%). Its `best_union`
stopped as soon as it found a set of documents with no internal step overlap,
which is not the same as the largest union: three documents covering {1}, {2}
and {1,3} have no-overlap pairs worth 2 steps, while the best pair covers 3.
The table above is an exact recomputation: dynamic programming over the
subsets of each query's step set (`dp[mask]` = fewest documents covering
`mask`), so the value is the true maximum for every query. The reviewer's
independent exhaustive recomputation agrees to four decimals (1.3405 /
1.1844).

Both numbers are *relative changes in the count of covered steps*, not
coverage points and not a bound on nDCG gains.

## Reading

1. **At fixed relevance, complementary swapping is worth about 5.5%.** That
   is the headroom for a ranker that only reorders relevant documents among
   themselves; it is larger than the first version said, but still small, and
   it says nothing about gains that also require finding *more* relevant
   documents.
2. **The experiment cannot speak to the 138 zero-score queries.** Every query
   whose top ten contains no judged-relevant document contributes 0 by
   construction, even though 86 of them do have a gold document inside the
   dense top-100. Improving those queries changes both relevance and coverage
   at once, and is outside this counterfactual.
3. **Where the gap actually sits.** The top ten covers 61% of annotated steps
   (1.2702/2.0739); the pool makes 82% reachable (1.6967/2.0739). Closing the
   0.43-step gap means ranking gold documents that are already in the pool
   higher, which is what nDCG@10 rewards (pool oracle: 0.7446 nDCG@10 at
   1.6967 steps). At this granularity the two counts move together, but this
   identity is a property of this fixed-slot construction, not a theorem.
4. **Genuinely separate headroom: the pool.** 0.377 steps per query (18% of
   annotations) never enter the top-100 the reranker sees.

## Consequence for the plan

- Priority 1 (hard-pair supervision) should be tested as a *re-weighting of
  the listwise objective*: its value is whether it ranks pool golds higher,
  and that must show up in nDCG@10.
- Priority 2 (fixed-budget candidate remediation) has a well-defined ceiling
  of 0.377 steps per query and 18% of annotations; that is the number to beat.
- The reported step-coverage metric (59.5% fused / 51.4% cross-encoder alone)
  should be read as "how much of the annotated answer the top ten shows",
  which is 61% by this per-query count, not as a deficit a reordering layer
  can repair.
