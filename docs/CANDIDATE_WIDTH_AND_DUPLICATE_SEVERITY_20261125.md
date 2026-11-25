# Candidate width and duplicate severity in the 8B fallback

The training-selected rule increases official dev domain-macro nDCG@10 from **0.39190 to 0.39227** on all 519 queries. It keeps the 34-document shortlist on the 263 queries where the six-pair policy abstains and reuses the existing 8B scores. Estimated model prompts remain 54.43 per query. Membership of the accepted branch and its ranking policy follow the [prior replay](REDUNDANCY_ACCUMULATION_20261125.md).

## Moving duplicate checks into candidate selection

The baseline keeps the ten frozen results and the first 24 dense outsiders. The novelty filter retains the ten and can skip outsiders whose query-matched window overlaps a window already in the shortlist. Deferred outsiders fill any remaining slots. Two 34-document variants protect either zero or 12 dense outsiders before applying the filter. The 58-document filtered variant protects 24. All arms rerank with the same maximum-similarity penalty. Frozen and 8B scores are z-standardized within each arm's own shortlist.

| Abstention candidate pool | 65-query 8B holdout | 519-query official dev | Dev candidate oracle change | Extra prompts per dev query |
|---|---:|---:|---:|---:|
| Dense, 34 | 0.30969 | **0.39190** | +0.00000 | 0.00 |
| Novelty filter, 34, retain 12 outsiders | 0.31193 | 0.39105 | +0.00123 | 0.00 |
| Novelty filter, 34 | 0.31204 | 0.39116 | +0.00265 | 0.00 |
| Dense, 58 | 0.31155 | 0.39031 | +0.03075 | 24.32 |
| Novelty filter, 58, retain 24 outsiders | 0.31249 | 0.39091 | +0.03054 | 24.32 |
| Dense, 100 | 0.31521 | 0.39039 | +0.05880 | 66.89 |

Oracle changes use ideal ordering inside the given candidate set and zero change on accepted queries. The wider shortlist finds more labeled documents and simultaneously adds competitors and changes local score normalization. In this comparison, those additional candidates yield a larger oracle gain than realized score gain. Cross-fitting shortlist selection on the 523-query frozen-score proxy gives a change of −0.00005 for both the equal-budget and unrestricted rules. [Candidate replay and paired results](../results/analysis/novel_candidate_pools_20261125.json), [full-system summary](../results/analysis/novel_candidate_pool_summary_20261125.json)

## Increase the penalty only when two windows are almost identical

Keep the 34 candidates. Each passage is represented by contiguous five-token sequences. The original rule uses the largest Jaccard overlap with an already selected passage, counting overlaps of at least 0.8. The training-selected rule doubles the contribution from overlaps of at least 0.95 and retains the original contribution on the rest. Each selection step subtracts the largest resulting penalty from the blended score. The ranking still starts with the highest blended score.

Choose the rule on the same 523 out-of-fold training queries used as a frozen-score proxy, disjoint from the 65 actual 8B holdout and the dev set. The pooled proxy rises from 0.26978 to 0.27763. Its three-fold held-out selection picks uniform 2.0 once and the 0.95-specific boost twice. The held-out pooled gain is +0.00627 with a paired within-domain 95% interval [+0.00327, +0.00787] and a domain-resampling interval [−0.00404, +0.02275]. The largest proxy contribution comes from IOTA. [Training-only rankings](../results/analysis/redundancy_stronger_proxy_20261125.json)

| Penalty | 65-query 8B holdout | 519-query official dev |
|---|---:|---:|
| Uniform 1.0, reference | 0.30969 | 0.39190 |
| Uniform 1.5 | 0.30993 | 0.39177 |
| Uniform 2.0 or 3.0 | 0.30993 | 0.39246 |
| Double overlap ≥0.95, selected by pooled training proxy | **0.30993** | **0.39227** |
| Triple overlap ≥0.95 | 0.30993 | 0.39227 |

The selected rule improves 14 dev queries and lowers two. The full-dev increase is **+0.00037**, with a within-domain paired 95% interval [−0.00002, +0.00097] and a domain-resampling interval [+0.00003, +0.00091]. Bitcoin contributes the largest domain gain. Domain-macro annotated-step coverage on queries with step labels changes from 62.99% to 63.07%. The separate 65-query 8B holdout adds +0.00023 on two queries. The official dev set had already informed earlier experiments. It serves as a replay evaluation here. Uniform 2.0 has the highest observed dev score, while the pooled training proxy selects the targeted rule and resolves the tie between the two targeted arms in favor of the smaller boost. [8B replay, per-domain effects and intervals](../results/analysis/redundancy_stronger_replay_20261125.json)

At the same prompt budget, near-identical passages benefit from a stronger penalty than ordinary near-duplicates. The official dev gain is small. The combination of wider-pool and fixed-pool controls locates it in the order of the retrieved documents and gives a margin and redundancy diagnostic for future rerankers.

## Reproduction

Run the proxy and 8B commands in [the original penalty study](REDUNDANCY_ACCUMULATION_20261125.md) with `--stronger`. Use fresh output paths and pass the stronger proxy to `--proxy-report`. The 8B command reuses archived candidate scores.

```bash
python scripts/replay_novel_candidate_pools.py --data-root data/track1_tempo \
  --proxy results/analysis/redundancy_disjoint_proxy_20261125.json \
  --oof8b results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev8b results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --dev-reference results/analysis/position_oracle_dev_20261124.json \
  --output results/analysis/novel_candidate_pools_replay.json

python scripts/summarize_novel_candidate_pools.py --data-root data/track1_tempo \
  --replay results/analysis/novel_candidate_pools_replay.json \
  --prior results/analysis/redundancy_accumulation_20261125.json \
  --protected results/analysis/accepted_weight_full_dev_audit_20261125.json \
  --proxy results/analysis/redundancy_disjoint_proxy_20261125.json \
  --output results/analysis/novel_candidate_pool_summary_replay.json
```
