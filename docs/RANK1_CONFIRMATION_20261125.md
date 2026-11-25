# When to replace the first result after an accepted rerank

## Fixed confirmation rules

The reference policy keeps rank one from 30% 8B fusion and reorders ranks 2–10 at 60%. On an accepted query where unrestricted 60% proposes a different leader, choose between the protected order and the unrestricted order. The documents in the top ten are fixed. Keep the 34-document abstention branch from the 0.38770 policy. [Rank-one protection and source scores](ACCEPTED_WEIGHT_FULL_DEV_20261125.md)

Evaluate rank-one proposals by the frozen ranker, the standalone 8B score, and the 4B model. The pair confirmation asks whether the 4B score of the proposed leader exceeds its score of the incumbent. The list confirmation asks whether the 4B 60% fusion picks the same leader. Both use cached scores against the same reference passage. The estimated extra cost applies only when unrestricted 8B proposes a new leader. Pair confirmation would score at most two documents in two display orders.

| Rank-one rule | Complete policy on training holdouts | Official dev | Helpful / harmful / neutral changes on dev | Estimated extra 4B prompts per dev query |
|---|---:|---:|---:|---:|
| Protect rank one | **0.31102** | 0.38770 | 0 / 0 / 0 | 0 |
| Proposed leader in the frozen top three | 0.30672 | 0.38702 | 4 / 5 / 14 | 0 |
| Proposed leader first by the raw 8B score | 0.29796 | 0.39201 | 16 / 17 / 40 | 0 |
| Both frozen top three and raw 8B first | 0.30672 | 0.38760 | 2 / 2 / 9 | 0 |
| 4B pair verification | 0.30144 | 0.38988 | 16 / 17 / 37 | 0.62 |
| 4B list verification | 0.29855 | 0.38706 | 5 / 6 / 17 | 3.05 |
| Always use the unrestricted leader | 0.29796 | 0.39193 | 20 / 20 / 47 | 0 |
| Choose the better of the two lists using qrels | 0.31524 | 0.39636 | 20 / 0 / 0 | Diagnostic |

Selecting on the pooled 65 training holdouts chooses protection for both the free rules and the 4B-assisted rules. Cross-fit selection alternates between protection and a permissive choice, yielding 0.29722 without 4B and 0.30070 with 4B, compared with 0.31102 for protection. On dev, the pair confirmation's +0.00218 increment has paired interval [−0.00250, +0.00697]. The oracle over the two existing choices quantifies the opportunity from finding the beneficial proposals. It uses the evaluation labels and serves as a diagnostic ceiling for this action set. [Rules and per-query decisions](../results/analysis/rank1_confirmation_20261125.json)

## Calibrate the rule on the independent 180-query 4B sample

Fit a domain-weighted ridge predictor to the observed nDCG change of releasing rank one. The 180-query training sample contains 49 candidate first-position changes and is disjoint from the 65 earlier holdouts. The three inputs are the challenger-minus-incumbent score difference for each standardized score source and the incumbent's new rank if released. The regularization coefficient is 0.1. Produce leave-one-domain-out training predictions and select among releasing for estimated gains above zero, above 0.01, or protecting every query.

Leave-one-domain-out validation selects the zero threshold, with +0.00012 relative to protection on 4B. Applying the same model and threshold to the 8B holdouts gives **−0.01246**, with paired interval [−0.02505, 0]. Official dev gives **0.38980**, +0.00209 [−0.00101, +0.00558]. The model approves three swaps on the earlier holdouts and eighteen on dev. These results add a transfer test of the learned margin. The more conservative rank-one policy retains the stronger evidence across the measured splits. [Training sample and inferred scores](ACCEPTED_WEIGHT_TRANSFER_20261125.md) · [Transfer results](../results/analysis/rank1_confirmation_20261125.json)

## Reproduction

```bash
python scripts/replay_rank1_confirmation.py --data-root data/track1_tempo \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --previous results/analysis/accepted_weight_full_dev_audit_20261125.json \
  --oof-8b results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --oof-4b results/analysis/anchor_reranker_oof_sample26_20261124.json results/analysis/anchor_reranker_oof_replication39_20261124.json \
  --dev-8b results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --dev-4b results/analysis/accepted_weight_full_dev_qwen4b_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --train-4b results/analysis/accepted_weight_transfer_qwen4b_20261125.json \
  --output results/analysis/rank1_confirmation_replay.json
```
