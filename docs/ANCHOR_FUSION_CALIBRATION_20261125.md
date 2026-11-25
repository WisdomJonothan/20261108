# Fusion calibration at a fixed 8B inference budget

Use the same 10-document accepted lists, 34-document abstention pools, and cached 8B scores as the 0.38374 replay. Keep the normalization for each route from that experiment. All policies use the same estimated 54.43 prompts per dev query. [Budget and score construction](BUDGETED_ANCHOR_FALLBACK_20261125.md)

## Training and controls

For score fusion, select the 8B weight from 0.1 to 0.6 in steps of 0.1 either jointly or separately by route. A rank-fusion control sums reciprocal ranks with weights 0.7 and 0.3 and offset 60. A pairwise logistic ranker learns nonnegative coefficients for the two standardized scores and standardized reciprocal dense rank. Its L2 penalty towards the existing [0.7, 0.3, 0] coefficients is fixed at 0.10. Compare uniform positive-negative pairs with pairs weighted by the absolute difference in top-ten rank discounts under the original model. For each query, divide weights by their sum. Give equal weight to represented training domains after removing queries with zero usable pairs.

Fit on one training cohort and score the other in both directions (26 and 39 queries). Refit on all 65 training holdouts for official dev. Read official dev qrels for evaluation.

| Same-budget policy | Cross-fit training holdouts | Official dev |
|---|---:|---:|
| Fixed 70:30 | 0.30821 | **0.38374** |
| One grid-selected weight | 0.30380 | **0.38374** |
| Separate grid-selected weights | **0.31709** | 0.37433 |
| Weighted RRF | 0.28894 | 0.37607 |
| Pairwise fit with uniform weights | 0.29957 | 0.38284 |
| Pairwise fit with rank discounts | 0.30074 | 0.38273 |

The shared fit selects 30% 8B on the pooled holdouts. The separate fit selects 30% for accepted lists and 60% for abstentions in both cross-fit directions and after pooling. This raises the cross-fit score by +0.00888 and lowers official dev by −0.00941 [−0.01834, −0.00103]. Weighted RRF also loses on dev with a paired interval [−0.01332, −0.00225]. [Coefficients, scores, and paired results](../results/analysis/anchor_fusion_calibration_20261125.json)

## Where the pairwise fit helps and hurts

The position-weighted fit improves accepted-list ordering by **+0.00862** on dev [+0.00095, +0.01773] and lowers the abstention route by **−0.00963** [−0.01862, −0.00128]. These are contributions to the total 13-domain macro. Eight domains improve and five decline on the accepted route. Leaving out any one domain gives a positive accepted-route contribution from +0.00557 to +0.01129. The two contributions nearly cancel when both routes are used.

Using the position-weighted fit on accepted lists alone is a retrospective diagnostic. It scores **0.39236** on dev and contributes **−0.01691** to the held-out training macro [−0.03329, −0.00237]. The dev-only gain needs independent confirmation before this rule serves as the submission policy.

The fitted accepted-route dense coefficient is zero, and its 8B coefficient represents about 59% of the sum of its coefficients. The fixed grid gives +0.00819 on dev from changing the 8B weight to 60% on accepted lists, compared with −0.01025 on held-out training. Most of the observed dev gain can be reproduced by changing the weight of the existing scores. The held-out and dev signs still disagree. [Per-route ablations and complete fixed-weight curves](../results/analysis/anchor_fusion_calibration_20261125.json)

## Reproduction

```bash
python scripts/replay_anchor_fusion_calibration.py --data-root data/track1_tempo \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --oof results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --output results/analysis/anchor_fusion_calibration_replay.json
```
