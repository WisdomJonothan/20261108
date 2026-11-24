# Choosing which queries receive a deeper candidate pool

The [zero-query audit](POSITION_REPAIR_20261124.md) locates remaining relevant material beyond dense depth 100. The [depth oracle](SWAP_EVIDENCE_SPANS_20261125.md) shows that depth 300 increases the ideal ranking score. This audit tests where to allocate that additional depth when the number of expansions is fixed.

Keep the dense top 100 and extend selected queries to top 300. Match the number of expanded queries within every domain to the number of times the six-pair judge abstained. This allocates 263 expansions across 519 queries and scores a mean of 201.35 candidate documents if the rule is executed. Compare the existing abstentions with the same-budget queries having smallest dense top-ten margin or smallest overlap between the frozen-fusion and dense top tens. Use the gap between dense ranks one and ten divided by the score standard deviation over the first 100 documents. Higher rank disagreement breaks ties using the dense-score margin, then query ID. Each selected candidate list is nested within the corresponding dense run.

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


## Train-only supervised routing control

Cross-fit on the 969 existing training-fold queries. Each fit uses two folds for learning and the third for evaluation. A pooled fit from the training folds then scores all 519 official dev queries. The five features are dense rank-one to rank-ten score margin, the dense head-to-tail score gap, dense/fused top-ten overlap, mean dense rank of the fused top ten, and the six-pair abstention indicator. Normalize features using only the fit sample and weight domains equally. Ridge regression predicts the ideal nDCG gain. Logistic regression predicts the first judged positive becoming reachable between depth 100 and 300. Fix the two regularization settings and use the same expansion quota per domain for all methods. The OOF quota is 553 queries and the official dev quota is 263.

| Rule | OOF ideal gain | OOF newly reachable zeroes | Dev ideal gain | Dev newly reachable zeroes |
|---|---:|---:|---:|---:|
| Six-pair abstention | +0.06808 | 35 | +0.04889 | 9 |
| Dense/fused rank disagreement | **+0.06858** | 35 | **+0.06401** | 16 |
| Ridge estimate of ideal gain | +0.06501 | 28 | +0.06154 | 13 |
| Logistic estimate of first-positive recovery | +0.06614 | 32 | +0.05411 | 12 |

On dev, ridge changes the disagreement score by −0.00247 [within-domain paired 95% interval −0.01866, +0.01685]. Logistic regression changes it by −0.00991 [−0.03323, +0.01341]. Rank disagreement gives the best point estimate of ideal gain in both splits among these measured routing rules. It is the straightforward candidate for a later same-judge test. The training-fold sample contains 52 zeroes with a first positive appearing between depths 100 and 300, compared with 26 in official dev.

[Train-fold and official-dev paired results](../results/analysis/learned_candidate_depth_20261125.json)

```bash
python scripts/fit_candidate_depth_routing.py --runs-root runs --data-root data/track1_tempo --output results/analysis/learned_candidate_depth_replay.json
```


## Candidate depth and edit limits

Join the same per-query routes with the [exact edit-budget audit](SWAP_EVIDENCE_SPANS_20261125.md). Each column uses optimal choices with respect to the official labels and the six-pair starting list. The single-admission reorder column permits reordering the selected ten, and unrestricted ranking permits any top ten from the chosen pool.

| Candidate depth policy | One in-place replacement | Two in-place replacements | One admission plus reordering | Unrestricted |
|---|---:|---:|---:|---:|
| All at depth 100 | 0.58504 | 0.62680 | 0.69850 | 0.74462 |
| Expand abstentions | 0.62964 | 0.67794 | 0.73585 | 0.79352 |
| Domain-matched uniform expansion, expectation | 0.63116 | 0.68219 | 0.73705 | 0.79694 |
| Expand rank disagreement | 0.63798 | 0.69080 | 0.74463 | 0.80864 |
| All at depth 300 | 0.67603 | 0.73982 | 0.77450 | 0.85132 |

Against a target of 0.60, one in-place replacement at depth 100 is bounded by 0.58504. Under the selective depth-300 route, the same edit allowance has a ceiling of 0.63798. Reordering the selected ten at depth 100 already raises the one-admission ceiling to 0.69850. The two mechanisms provide separate headroom estimates. All table scores are label-based diagnostic ceilings.

[Joined depth and edit-budget estimates](../results/analysis/adaptive_candidate_depth_edit_limits_20261125.json)

```bash
python scripts/audit_depth_edit_limits.py --ceilings results/analysis/competition_edit_budget_20261125.json --routing results/analysis/adaptive_candidate_depth_dev_20261125.json --output results/analysis/adaptive_candidate_depth_edit_limits_replay.json
```


## Remaining zero-score queries after the 8B evaluation

Join the 519 official-dev results to the pre-existing routes by domain and query ID. The six-pair reference, direct depth-100 8B blend and selective 8B replay have 130, 129 and 125 zero-score queries, respectively. Of these, 78, 77 and 73 already have judged positives at depth 100. Each ranking has the same 52 zero-score queries with no judged positive at depth 100. The respective totals describe two targets for the next experiment: improving first-page ranking for queries with retrieved positives and retrieving the first positive for the remaining queries.

At the fixed budget of 263 expansions, rank disagreement makes 16 of the 52 misses reachable, versus nine with the abstention route. Across all 519 queries, perfect ranking of only those newly reachable queries would contribute +0.03168 for disagreement and +0.02163 for abstention. These ceilings sum each selected first-positive query's depth-300 oracle within its original domain and give zero gain to all other queries. [Original per-query depth and routing audit](../results/analysis/adaptive_candidate_depth_dev_20261125.json) · [Audited 8B scores per query](../results/analysis/anchor_official_dev_qwen8b_20261125.json)
