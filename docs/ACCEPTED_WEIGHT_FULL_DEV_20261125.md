# Verify judge weighting across every accepted list

## Protecting the first result

Replay Qwen3-8B scores for the same accepted-list membership and the same normalization as the 0.38374 system. Preserve the first k positions from 30% judge fusion. Order the remaining candidates by the 60% fusion. Preserve the original 34-document rule on all abstentions. The reference and every arm use the same estimated 54.43 prompts per official dev query. [Fixed-pool protocol](BUDGETED_ANCHOR_FALLBACK_20261125.md)

| First positions held at the 30% result | Whole policy on training holdouts | Whole policy on official dev |
|---|---:|---:|
| 0 | 0.29796 | 0.39193 |
| 1 | **0.31102** | **0.38770** |
| 3 | 0.30831 | 0.38391 |
| 10 | 0.30821 | 0.38374 |

Choose among these four scopes on the 65 training holdouts. Both directions of fitting on the 26-query and 39-query cohorts and scoring the other select k = 1. On the full 519-query dev denominator, rank-one protection gains **+0.00396** [+0.00005, +0.00811] over the 30% baseline. Resampling domains gives [−0.00213, +0.00945]. The cross-fit training gain is **+0.00281** [−0.00151, +0.00736]. These are cached replays.

Under unrestricted 60%, rank one changes on 13 of the 35 accepted training queries and 87 of the 256 accepted dev queries. On the accepted subsets, the rank-one contribution is −0.04861 on training and +0.01973 on dev, with substantial offsets at other positions. Protecting the first result removes this source of reversal and yields positive full-policy point estimates in both samples. On the new 180-query 4B training subset from the prior experiment, the same protection changes 0.28789 to 0.28770, a difference of −0.00018 [−0.00569, +0.00534]. [Prior score-transfer experiment](ACCEPTED_WEIGHT_TRANSFER_20261125.md)

## Fresh 4B inference on all 256 accepted dev queries

Rescore the complete accepted subset with Qwen3-4B-Instruct-2507. Keep the same ten documents and the same rank-five reference for each model. The fresh 4B arm uses BF16, both display orders, and clipping only for words of 2,048 or more characters in the selected 112-word window. The 8B arm uses the original full-depth score cache, with its original prompt batching. The 4B run scores 4,662 prompts and 2,600,162 input tokens in 430 seconds on the RTX 3090, with 10,286 MiB peak PyTorch allocation. The original 8B weight directory is unavailable on the server, and the pinned checkpoint URL also times out from the Windows host.

| Judge on the same 256 queries | Fixed 30% | Fixed 60% | Protect first at 30%, reorder rest at 60% |
|---|---:|---:|---:|
| Fresh 4B | 0.33213 | 0.32634 | **0.33817** |
| Archived 8B | 0.32885 | **0.34316** | 0.33630 |

These are conditional macro scores over all 13 domains, using only accepted queries in each domain. The difference between the 60%-versus-30% effects of 8B and 4B is **+0.02010** [within-domain +0.00396, +0.03787, domain bootstrap +0.00403, +0.03769]. With rank-one protection, both models improve over their own 30% controls on accepted dev lists, by +0.00604 [+0.00054, +0.01198] for 4B and +0.00745 [+0.00162, +0.01348] for 8B.

On the original matched 80 queries, freshly scored 4B gives −0.01477 for 60% versus 30%, close to the archived −0.01429. On the newly covered 176 queries, the point estimate is +0.00234 for 4B [−0.00839, +0.01233] and +0.01503 for 8B [+0.00006, +0.03161]. The larger comparison confirms the overall model-by-weight interaction on these dev lists and shows how the 80-query filter changed its size. [Fresh 4B scores](../results/analysis/accepted_weight_full_dev_qwen4b_20261125.json) · [Matched scores and per-position attribution](../results/analysis/accepted_weight_full_dev_audit_20261125.json)

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_accepted_weight_transfer.py --split dev \
  --reference results/analysis/position_oracle_dev_20261124.json \
  --model models/Qwen3-4B-Instruct-2507 --checkpoint /tmp/reteco-accepted-dev-4b \
  --output results/analysis/accepted_weight_full_dev_qwen4b_replay.json

python scripts/audit_accepted_weight_changes.py --data-root data/track1_tempo \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --oof results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --fresh-4b results/analysis/accepted_weight_full_dev_qwen4b_replay.json \
  --fresh-train-4b results/analysis/accepted_weight_transfer_qwen4b_20261125.json \
  --previous-match results/analysis/accepted_weight_checkpoint_comparison_20261125.json \
  --output results/analysis/accepted_weight_full_dev_audit_replay.json
```
