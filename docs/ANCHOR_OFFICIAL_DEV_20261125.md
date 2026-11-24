# Qwen3-8B at depth 100 on official dev

The [held-out checkpoint comparison](ANCHOR_EDIT_SCOPE_20261125.md) found that the 8B shared-reference judge benefits from allowing changes beyond one fixed-position replacement. The full official-dev run fixes the same Qwen3-8B checkpoint, non-thinking template, 112-word windows, rank-five reference, both display orders, and 70:30 fusion with the frozen system. Every query uses its fixed dense top 100. The main comparison is against the existing six-pair submission configuration.

## Main official results

All scores are domain-macro nDCG@10 over 519 queries and 13 domains. Evidence coverage averages positive labeled steps within the 484 annotated queries and then domains.

| System | nDCG@10 | Evidence coverage | Judge prompts per query |
|---|---:|---:|---:|
| Frozen dense + fine-tuned BGE | 0.35810 | 59.26% | 0 |
| Six-pair competition reference | 0.37350 | 61.39% | 12 |
| Qwen3-4B protected depth-20 blend, direct | 0.36817 | 61.17% | 38 |
| Qwen3-8B protected depth-20 blend, cached subset | 0.37266 | 60.96% | 38 if scored directly |
| Qwen3-8B depth-100 score alone | 0.30498 | 57.96% | 198 |
| Qwen3-8B depth-100 blend, direct | **0.37959** | 62.26% | 198 |
| Six-pair with depth-100 8B blend on abstention, replay | **0.38206** | 62.62% | 112.34 if scored selectively |

The direct depth-100 blend gains **+0.02149** over frozen fusion, with paired within-domain 95% interval [+0.01048, +0.03265] and domain-resampling interval [+0.01356, +0.03031]. All thirteen domains improve. Relative to the six-pair competition reference it gains **+0.00609** [−0.00803, +0.01987], with domain-resampling interval [−0.00355, +0.01524]. Nine domains improve and four decline. The 20-document 8B replay holds the original top ten, fills with the highest dense outsiders, and recomputes normalization within that subset. Using all 100 improves on this replay by **+0.00693** [+0.00075, +0.01356]. The direct inference used the entire pool. The depth-20 score comes from a subset of that score cache.

The abstention replay retains the competition output when the six-pair policy accepts an action. It applies the fixed depth-100 blend to the other 263 queries. It gains **+0.00855** [+0.00046, +0.01733] relative to six-pair, with domain-resampling interval [+0.00233, +0.01507]. This secondary result improves 98 queries and lowers 43, recovering nine competition zeroes and creating four. A qrels-based choice between the direct 8B blend and six-pair would reach 0.40505 on the same outputs.

## Where the direct score gain comes from

| Edit scope on the same depth-100 scores | nDCG@10 |
|---|---:|
| Start from the six-pair result | 0.37350 |
| Reorder the same ten | 0.37994 |
| Admit one document in the vacated slot | 0.37290 |
| Keep that admission and sort the same ten | 0.38082 |
| Unrestricted blend ranking | 0.37959 |

The two single-admission arms contain the exact same documents for every query. Permitting reorder adds **+0.00793** [−0.00194, +0.01786]. Allowing more admissions after reorder changes the point estimate by −0.00123 [−0.00490, +0.00217]. The direct unrestricted blend moves 72 judged relevant documents into the top ten and removes 43. Exact discounted attribution gives +0.01461 for additions, −0.01543 for removals, and **+0.00691** for rank changes among documents kept in both lists. The largest positive point estimate in the fixed-membership contrast comes from moving documents already present. The direct blend recovers twelve competition zeroes and introduces eleven. Future deeper retrieval can target the remaining candidate misses and separately audit front-rank changes.

## Runtime and recovery

The run uses 102,762 prompts and 59,314,056 input tokens. Successful scoring takes 4 hours 28 minutes on a single RTX 3090 with 21,965 MiB peak PyTorch allocation. After a CUDA memory failure, the resumable evaluator verifies and reloads 316 completed query checkpoints. Of the 519 queries, 516 use the original batch size of 12, one uses 6 and two use 1. The checkpoint, scoring precision and original prompt text are preserved. The direct gain over competition on the 516 nominal-batch queries is +0.00644. [Checkpoint verification and effective batches](../results/analysis/anchor_dev100_checkpoints_20261125.json) · [Prompt length audit](ANCHOR_PROMPT_LENGTH_20261125.md)

[Original 8B scores](../results/analysis/anchor_reranker_qwen8b_dev100_20261125.json) · [Official paired audit](../results/analysis/anchor_official_dev_qwen8b_20261125.json) · [Edit-scope replay](../results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json) · [Depth-20 replay](../results/analysis/anchor_edit_scope_qwen8b_dev20_replay_20261125.json)

The [oversized-input control](ANCHOR_PROMPT_LENGTH_20261125.md) rescores the 84 affected prompts after clipping the long words and reproduces the same official score. The [temporal contrast diagnostic](TEMPORAL_QUERY_CONTRAST_20261125.md) isolates temporal preference changes on reviewed pairs using the same 8B judge and a matched 4B prompting control.

## Reproduction

Start with the fixed inputs and a fresh checkpoint directory. Retry at a smaller per-query batch if needed. The evaluation writes document scores atomically and verifies the cached configuration when using the same command again.

```bash
python scripts/run_anchor_with_oom_retry.py --model models/Qwen3-8B --disable-thinking --split dev --pool-depth 100 --batch-size 12 --checkpoint-dir /tmp/reteco-anchor-dev100-checkpoints --output results/analysis/anchor_reranker_qwen8b_dev100_20261125.json

python scripts/summarize_anchor_checkpoints.py --checkpoint-dir /tmp/reteco-anchor-dev100-checkpoints --direct-report results/analysis/anchor_reranker_qwen8b_dev100_20261125.json --output results/analysis/anchor_dev100_checkpoints_replay.json

python scripts/replay_anchor_edit_scope.py --data-root data/track1_tempo --pool-scores results/analysis/anchor_edit_scores_dev100_20261125.json --inputs results/analysis/anchor_reranker_qwen8b_dev100_20261125.json --reference results/analysis/position_oracle_dev_20261124.json --starts frozen competition abstention --output results/analysis/anchor_edit_scope_qwen8b_dev100_replay.json

python scripts/replay_anchor_edit_scope.py --data-root data/track1_tempo --pool-scores results/analysis/anchor_edit_scores_dev100_20261125.json --inputs results/analysis/anchor_reranker_qwen8b_dev100_20261125.json --reference results/analysis/position_oracle_dev_20261124.json --protected-depth 20 --starts frozen abstention --compare results/analysis/anchor_edit_scope_dev20_20261125.json --output results/analysis/anchor_edit_scope_qwen8b_dev20_replay.json

python scripts/audit_anchor_dev.py --data-root data/track1_tempo --raw results/analysis/anchor_reranker_qwen8b_dev100_20261125.json --full results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json --replay20 results/analysis/anchor_edit_scope_qwen8b_dev20_replay_20261125.json --control20 results/analysis/anchor_edit_scope_dev20_20261125.json --recovery results/analysis/anchor_dev100_checkpoints_20261125.json --output results/analysis/anchor_official_dev_qwen8b_replay.json
```
