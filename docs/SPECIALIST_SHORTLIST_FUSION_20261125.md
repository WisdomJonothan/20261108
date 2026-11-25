# Add a specialist relevance score on the fixed 34-document pool

Use the same training holdouts and official dev query IDs as the budgeted 8B replay. Score every document in the 34-candidate abstention pools with Qwen3-Reranker-4B in BF16. Use the official yes/no prompt with its assistant suffix and the same selected 112-word passages. Clip individual words of at least 2,048 characters to 256 characters. Keep the yes-minus-no logit as the score. This preserves differences between high-confidence decisions that can round to the same probability. The original 8B score is taken from the existing cache and the accepted-list branch stays fixed for the contrast. [Prior budgeted protocol](BUDGETED_ANCHOR_FALLBACK_20261125.md) · [Help and harm in the earlier specialist pair control](SWAP_JUDGE_CHECKPOINT_20261124.md)

Each fusion begins with the two existing z-normalized scores at 0.7 and 0.3. For the first calibration family, transfer some of the 8B weight to the pointwise specialist. The second family adds specialist weight to the fixed blend. Consider weights 0.075, 0.15, 0.225 and 0.3. Compare these with regularized nonnegative pairwise fits, with and without weighting pairs by rank discount. Select the policy in both directions across the 26- and 39-query cohorts and refit on all 65 to score dev. The cached 8B baseline is included in the choices.

## Results

The specialist scores 9,962 query–passage pairs, including 1,020 training pairs and 8,942 dev pairs. The run processes 4,502,835 tokens in 824 seconds on one RTX 3090 with a peak of 9,956 MiB PyTorch allocation. Any deployed use of all specialist scores on this route would require 17.23 additional prompts per official dev query on average. The fixed 8B reference needs 54.43 estimated prompts per query.

| Method | Cross-fit training nDCG@10 | Official dev nDCG@10 |
|---|---:|---:|
| Fixed original 8B policy | 0.31102 | **0.38770** |
| Add specialist weight of 0.075 | 0.30895 | 0.38503 |
| Add specialist weight of 0.15 | 0.30658 | 0.38608 |
| Replace half of the 8B weight with specialist | 0.29557 | 0.38260 |
| Uniform pairwise fit | 0.31531 | 0.38291 |
| Rank-weighted fit selected on training | **0.31570** | 0.37807 |
| Selected fit plus the prior duplicate penalty | 0.31961 | 0.38004 |

All eight static ways of adding or substituting a specialist weight produce lower official dev point estimates. Both pooled pairwise fits assign the specialist coefficient zero. The rank-weighted fit's dev decrease of −0.00963 [−0.01862, −0.00128] comes from the refitted original and 8B scores. The combined fit scores 0.38004 compared with 0.38824 for the training-selected near-duplicate penalty alone. On unchanged prompt budget, the near-duplicate result is the improved dev configuration. [Near-duplicate reference](SHORTLIST_REDUNDANCY_20261125.md) · [All model scores](../results/analysis/specialist_shortlist_scores_20261125.json) · [All fusion and combination results](../results/analysis/specialist_shortlist_fusion_20261125.json)

## Reproduction

Create the manifest following the first command in the near-duplicate report. Then score with the specialist.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/score_specialist_shortlists.py \
  --manifest /tmp/reteco-34-doc-manifest.json --model models/Qwen3-Reranker-4B \
  --checkpoint /tmp/reteco-specialist-shortlists \
  --output results/analysis/specialist_shortlist_scores_replay.json

python scripts/replay_specialist_fusion.py --data-root data/track1_tempo \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --protected results/analysis/accepted_weight_full_dev_audit_20261125.json \
  --scores results/analysis/specialist_shortlist_scores_replay.json \
  --oof-8b results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev-8b results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --redundancy results/analysis/shortlist_near_duplicates_20261125.json \
  --output results/analysis/specialist_shortlist_fusion_replay.json
```
