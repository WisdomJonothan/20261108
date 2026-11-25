# Adaptive shortlist width at the same 8B prompt budget

## Protocol

Start from the 0.38374 replay. On accepted six-pair queries, use the previous same-ten reorder. On six-pair abstentions, protect the original ten and score additional dense candidates, with both score sources standardized inside the candidate set. The previous policy always adds 24 candidates, giving 34 documents in total. Test two allocations with that same total prompt budget in every domain. For the wide allocation, trade two reductions from 24 to 12 against one increase from 24 to 48. For the narrow allocation, trade one reduction from 24 to 18 against one increase from 24 to 30. Use width 24 for the remainder within a domain.

Compare four signals for allocating bigger pools to queries. Smaller overlap with the original ten or a smaller score gap between ranks 10 and 11 indicates more uncertainty. Compute these quantities either upstream or from a 22-document 8B preview. The preview scores are a subset of every final candidate pool and count once. Each signal induces a batch allocation within its domain. Fix the signals and select one per allocation on the 65 held-out training queries.

## Same-budget scores

| Allocation | Number of dev abstentions at each width | Selected signal, held-out change | Selected signal, dev score | Uniform allocation, expected dev score | Label-based best allocation on dev |
|---|---|---:|---:|---:|---:|
| Fixed 24 extras | 263 at 24 | Baseline | **0.38374** | 0.38374 | 0.38374 |
| Wide 12 or 48 | 168 at 12, 11 at 24, 84 at 48 | Upstream overlap, +0.00000 | 0.38082 | 0.38179 | 0.38587 |
| Narrow 18 or 30 | 127 at 18, 9 at 24, 127 at 30 | Upstream overlap, +0.00082 | 0.38346 | 0.38327 | 0.38436 |

The paired dev differences for the wide and narrow allocations are −0.00293 [−0.00728, −0.00009] and −0.00029 [−0.00058, −0.00006], respectively. All four signals have lower point estimates than fixed width for both allocations. The four signals and the uniform control are evaluated on the same set of available rankings. The label-based allocation uses dynamic programming to select the best observed ranking for each query while satisfying the exact same per-domain quotas. Its gain over fixed width is +0.00213 for the wide allocation and +0.00062 for the narrow allocation.

The wide allocation selects only four held-out queries for width 48 because its per-domain budgets retain the middle width on 18 of the 30 training abstentions. Consider the held-out choice exploratory for this allocation. All 519-query scores use the 13-domain macro average. Each policy in this table averages the previously verified 54.43 prompts per query. [Fixed 34-document replay](BUDGETED_ANCHOR_FALLBACK_20261125.md) · [All allocations and scores](../results/analysis/adaptive_anchor_width_20261125.json)

## The same normalization change has different effects on the two routes

Use the frozen system's existing scores over all 100 documents to set its standard deviation. Compare this with setting that scale using the documents under consideration. Hold 8B normalization local for both choices. Centering contributes a query-wide constant to each source. The change that can affect ranking is the relative scale of the two score sources.

| Queries receiving the normalization change | Local documents | Median frozen local / full standard deviation on dev | Effect on total dev nDCG@10 |
|---|---:|---:|---:|
| Six-pair accepted | 10 | 0.702 | **+0.00308** [+0.00019, +0.00631] |
| Six-pair abstained | 34 | 0.964 | **−0.00144** [−0.00454, +0.00102] |

The paired difference between these two policy effects is +0.00452 [+0.00062, +0.00883]. Each effect uses the same 519-query denominator and counts zero on the unmodified route. The two routes select different queries and different-sized pools. For accepted lists, the smaller local frozen standard deviation typically amplifies differences in the original scores relative to the 8B scores. Full-pool scaling reduces that amplification. The 34-document fallback uses a wider pool and displays a different effect. These controls support treating normalization as part of each reranking policy. [Per-route normalization results](../results/analysis/adaptive_anchor_width_20261125.json)

## Reproduction

```bash
python scripts/replay_adaptive_anchor_width.py --data-root data/track1_tempo \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --oof results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --output results/analysis/adaptive_anchor_width_replay.json
```
