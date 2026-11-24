# Choosing which queries receive a deeper candidate pool

The [zero-query audit](POSITION_REPAIR_20261124.md) locates remaining relevant material beyond dense depth 100. The [depth oracle](SWAP_EVIDENCE_SPANS_20261125.md) shows that depth 300 increases the ideal ranking score. This audit tests where to allocate that additional depth when the number of expansions is fixed.

Keep the dense top 100 and extend selected queries to top 300. Match the number of expanded queries within every domain to the number of times the six-pair judge abstained. This allocates 263 expansions across 519 queries and scores a mean of 201.35 candidate documents if the rule is executed. Compare the existing abstentions with the same-budget queries having smallest dense top-ten margin or smallest overlap between the frozen-fusion and dense top tens. Use the gap between dense ranks one and ten divided by the score standard deviation over the first 100 documents. Higher rank disagreement breaks ties using the denser score margin, then query ID. Each selected candidate list is nested within the corresponding dense run.

## Results on the 519 official dev queries

The baseline depth-100 ideal nDCG@10 is 0.74462. All numbers in the right columns use official qrels over already retrieved lists. A newly reachable query has zero positives at depth 100 and at least one at depth 300.

| Expansion rule | Queries expanded | Ideal nDCG@10 gain | Newly reachable zero-score queries |
|---|---:|---:|---:|
| Six-pair abstentions | 263 | +0.04889 | 9 |
| Domain-matched uniform allocation, expectation | 263 | +0.05232 | 12.50 |
| Smallest dense margin | 263 | +0.05935 | 16 |
| Greatest dense/frozen ranking disagreement | 263 | +0.06401 | 16 |
| Best label-based expansion allocation | 263 | +0.10602 | 26 |
| Every query at depth 300 | 519 | +0.10670 | 26 |

At this matched budget, disagreement finds seven more newly reachable zeroes than abstention. The ideal score advantage over abstention is +0.01512 [within-domain paired 95% interval −0.00910, +0.03890]. The advantage over uniform allocation is +0.01169 [−0.00051, +0.02372]. The label-based allocation captures nearly all ideal depth-300 gain with about half the expansions. These estimates identify candidates for a controlled comparison with a fixed judge after the full depth-100 evaluation. The batch quotas use the evaluated queries' original decisions. A per-query deployment rule would set thresholds on training-fold data before official evaluation.

[Per-query routing, effective depth and paired results](../results/analysis/adaptive_candidate_depth_dev_20261125.json)

```bash
python scripts/audit_adaptive_candidate_depth.py --ceilings results/analysis/competition_edit_budget_20261125.json --reference results/analysis/position_oracle_dev_20261124.json --output results/analysis/adaptive_candidate_depth_dev_replay.json
```
