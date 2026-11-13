# Counterfactual coverage bound: what a coverage-aware reranker could get for free

Date: 2026-11-14. Script: `scripts/coverage_cf_bound.py`.
Outputs: `results/COVERAGE_CF_BOUND.json` (fused top ten), `results/COVERAGE_CF_BOUND_ce.json`
(cross-encoder's own top ten).

The third review asked for the counterfactual bound before committing to a
coverage-aware training objective: keep the relevance pattern of the top ten
fixed (the same judged-relevant documents at the same ranks, so nDCG@10 cannot
change), and only re-choose *which* relevant documents fill those slots, from
the pool the cross-encoder actually sees (dense top-100). If that swaps in more
annotated steps, a coverage-aware ranker has headroom at zero metric cost. If
it does not, coverage is not a separable lever.

Numbers are per-query averages of annotated steps covered in the top ten,
domain-macro over the 13 dev domains, 519 queries.

| ranking scored | steps in top ten | best with the same slots | pool ceiling | annotation ceiling | nDCG@10 |
|---|---|---|---|---|---|
| fused system | **1.2702** | 1.2989 | 1.6967 | 2.0739 | 0.3581 |
| cross-encoder alone | **1.1122** | 1.1593 | 1.6967 | 2.0739 | 0.2823 |

"best with the same slots" = for each query, hold the number of relevant
documents in the top ten fixed, and pick the most step-complementary set of
that size from the pool: 39 of 519 queries gain (44 step-slots) for the fused
list, 44 queries (58 slots) for the cross-encoder. That is **+0.029 and +0.047
steps per query, +2.3% and +4.2% relative** — the largest coverage gain
available without touching relevance.

## Reading

1. **Coverage-aware reordering is not a lever on its own.** At fixed nDCG the
   system leaves 2-4% of its coverage on the table. A loss term that only
   reorders relevant documents among themselves cannot pay for itself.
2. **The real gap is relevance, not step arithmetic.** The top ten covers 61%
   of annotated steps (1.2702/2.0739); the pool makes 82% reachable
   (1.6967/2.0739). Closing the 0.43-step gap between the two means ranking
   gold documents that are already in the pool higher — which is exactly what
   nDCG@10 rewards (oracle over the pool: 0.7446 nDCG@10 at 1.6967 steps).
   Coverage and relevance are therefore not separable objectives at this
   granularity: "cover another step" and "put another gold in the top ten" are
   the same event, seen under two names.
3. **The only genuinely separate headroom is the pool.** 0.377 steps per query
   (18% of annotations) never enter the top-100 the reranker sees; reaching
   them needs a deeper or better first stage, which is the candidate-side
   experiment, not a loss term.

## Consequence for the plan

- Priority 1 (hard-pair supervision) should be tested as a *re-weighting of
  the listwise objective*, not as a new "coverage mechanism": its value is
  whether it ranks pool golds higher, and that must show up in nDCG@10.
- Priority 2 (fixed-budget candidate remediation) has a well-defined ceiling
  of 0.377 steps per query and 18% of annotations; that is the number to beat.
- The reported step-coverage metric (59.5% fused / 51.4% cross-encoder alone)
  should be read as "how much of the annotated answer the top ten shows",
  which is 61% by this per-query count, not as a deficit a reordering layer
  can repair.
