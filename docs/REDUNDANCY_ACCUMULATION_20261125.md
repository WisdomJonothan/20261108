# Select the duplicate penalty on 523 disjoint training-fold queries

The prior replay tested duplicate penalties on a small 65-query 8B holdout and selected 0.25. The stronger 1.0 setting scored 0.39190 on official dev and was recorded as a sensitivity check. This follow-up selects the penalty with a larger, disjoint training-only run, then applies the selected rule to the fixed 8B shortlists. The prior official-dev results were known when planning this follow-up. [Original near-duplicate experiment](SHORTLIST_REDUNDANCY_20261125.md)

## Training-only selection

From the 969 out-of-fold query archive, take the six-pair abstentions and exclude every query from the original 65-query 8B holdout. This yields 523 queries across all 13 domains. For each query, keep the ten upstream documents and the first 24 dense outsiders. Use its out-of-fold frozen score, z-standardized within the 34 candidates and scaled by 0.7. This proxy omits the new 8B score and uses the same duplicate detection and ranking procedure.

Compare a maximum penalty based on the closest already selected window with penalties that accumulate similarity across selected windows. The capped variant limits accumulated similarity to two. Choose the best domain-macro rule on two folds and evaluate on the third. Repeat for all folds. Evaluate the pooled training-selected setting on 8B afterwards.

| Held-out training fold | Queries | Selected by the other folds | nDCG change on held-out fold |
|---|---:|---|---:|
| f0 | 175 | Maximum penalty 1.0 | +0.00645 |
| f1 | 175 | Maximum penalty 1.0 | +0.00539 |
| f2 | 173 | Maximum penalty 1.0 | +0.00349 |
| All, cross-fit | 523 | Maximum penalty 1.0 in every fold | **+0.00511** |

The paired 95% interval for the cross-fit proxy is [+0.00310, +0.00740], with 99 improved queries and five declined. The proxy scores 0.26467 at 0.25 and 0.26978 at 1.0. [Disjoint training-fold rankings](../results/analysis/redundancy_disjoint_proxy_20261125.json)

## Transfer to the fixed 8B shortlist

Reuse the original 34 candidates, the 8B scores, and the accepted-list policy. The estimated budget remains 54.43 prompts per official dev query. Each duplicate rule preserves the first document from the pre-penalty list.

| Penalty | Original 65-query 8B holdout | 519-query official dev |
|---|---:|---:|
| Maximum, 0.25 | **0.31144** | 0.38824 |
| Maximum, 0.5 | 0.31134 | 0.38894 |
| Maximum, 1.0, selected on the larger fold proxy | 0.30969 | **0.39190** |
| Accumulated, 0.25 | **0.31144** | 0.38915 |
| Accumulated, 0.5 | 0.31138 | 0.38922 |

Compared with 0.38824, the selected 1.0 penalty adds +0.00366 on dev, improving 43 queries and lowering eight. The paired within-domain 95% interval is [−0.00029, +0.00868]. The domain-resampling interval is [+0.00135, +0.00626]. The same setting loses 0.00174 on the original 65-query holdout. The IOTA contribution averaged across its own holdout queries is −0.02713. Omitting IOTA from that descriptive aggregation changes the held-out estimate to +0.00037. The larger frozen-score proxy provides independent query evidence for the stronger rule, while the 65-query 8B comparison records its transfer limitation. The default 0.25 accumulated rule matches the original holdout score and improves dev by +0.00091, with a paired interval close to zero. [Matched evaluation, proxy fold audit, and per-query rankings](../results/analysis/redundancy_accumulation_20261125.json)

## Reproduction

The proxy uses the original training-fold fused runs and official corpus. It requires CPU access only.

```bash
python scripts/replay_redundancy_accumulation.py --proxy --data-root data/track1_tempo \
  --reference results/analysis/position_oracle_oof_20261124.json \
  --exclude results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --output results/analysis/redundancy_disjoint_proxy_replay.json

python scripts/replay_redundancy_accumulation.py --data-root data/track1_tempo \
  --proxy-report results/analysis/redundancy_disjoint_proxy_replay.json \
  --duplicate results/analysis/shortlist_near_duplicates_20261125.json \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --oof results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --output results/analysis/redundancy_accumulation_replay.json
```
