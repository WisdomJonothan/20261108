# Reduce near-duplicate evidence in the 34-document fallback

## Method and metric

Use the 0.38770 policy with rank-one protection on accepted queries. On abstentions, retain the exact 34 candidates and original 70:30 fused scores. Represent the same lexical 112-word window with a set of consecutive five-word shingles. Treat two windows as near duplicates when both have at least 12 shingles and their Jaccard similarity is at least 0.8.

Construct the top ten greedily. At each rank, subtract the penalty coefficient times the largest retained similarity to any already selected document. For the first result, the penalty is zero. Compare the coefficients 0, 0.25, 0.5 and 1.0. Choose the coefficient on the 65 training holdouts before evaluating the official dev rankings. Candidate generation and all model-score calls are reused. The prompt budget is 54.43 per dev query. [Reference pipeline](ACCEPTED_WEIGHT_FULL_DEV_20261125.md)

| Near-duplicate penalty | Training holdouts | Official dev nDCG@10 | Dev paired 95% interval versus no penalty |
|---|---:|---:|---:|
| 0 | 0.31102 | 0.38770 | Reference |
| 0.25, selected on the training holdouts | **0.31144** | **0.38824** | [+0.00012, +0.00105] |
| 0.5 | 0.31134 | 0.38894 | [−0.00159, +0.00411] |
| 1.0 | 0.30969 | 0.39190 | [+0.00017, +0.00922] |

The selected penalty adds **+0.00054** on official dev and **+0.00042** on the pooled training holdouts. Its official-dev domain-resampling interval is [+0.00001, +0.00125]. It improves 30 dev queries, lowers five, and has a positive point estimate after omitting any one domain. The two training folds choose 1.0 and 0.25 when used alone. Carrying those fold-specific choices to the other fold yields 0.30959. The larger 1.0 dev result is a descriptive sensitivity check, with a lower pooled training score.

The near-duplicate detector fires on 244 of the 263 dev abstentions. The selected penalty changes the order of 128 lists and the membership of 62. The nDCG increase decomposes into +0.00034 from moving retained documents and +0.00020 net from document replacements. Annotated step coverage changes from 62.73% to 62.79%. It recovers three zero-score queries and creates one. [Per-query ranks and similarity edges](../results/analysis/shortlist_near_duplicates_20261125.json)

## Other cached controls

Mixing the existing 4B-Instruct judge into accepted-list tail reordering and transferring a heavier 8B tail weight to abstentions did not improve the pooled-selection dev result. Clipping the 8B z-score or adding a reward or penalty for disagreement also gave lower dev point estimates. [Accepted ensemble](../results/analysis/accepted_judge_ensemble_20261125.json) · [Abstention-tail controls](../results/analysis/abstention_tail_20261125.json) · [Robust score transforms](../results/analysis/robust_abstention_fusion_20261125.json)

## Reproduction

```bash
python scripts/score_specialist_shortlists.py --prepare \
  --oof results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --oof-scope results/analysis/anchor_edit_scope_qwen8b_oof100_20261125.json \
  --dev-scope results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json \
  --output /tmp/reteco-34-doc-manifest.json

python scripts/replay_shortlist_near_duplicates.py --data-root data/track1_tempo \
  --manifest /tmp/reteco-34-doc-manifest.json \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --protected results/analysis/accepted_weight_full_dev_audit_20261125.json \
  --oof8b results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev8b results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --output results/analysis/shortlist_near_duplicates_replay.json
```
