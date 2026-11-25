# Reduce the 8B reranking budget with protected shortlists

Replay the original two-order 8B document scores on two disjoint training holdouts (65 queries total) and all 519 official dev queries. When six-pair accepts a change, leave its ten documents in place for the main shortlist comparison. On an abstention, keep those ten and add 6, 12, 24, or 48 documents from the same dense top 100. Draw the extras either by dense rank, frozen-fusion rank, or RRF over those two ranks (offset 60). Recompute both z-scores inside each candidate pool and apply the fixed 70:30 combination. Use the held-out training score to choose the source at each width.

| Additional candidates on an abstention | Source selected on the 65 training queries | Held-out score | Official dev score | Mean prompts per dev query |
|---|---|---:|---:|---:|
| Six-pair only | No additional scoring | 0.26976 | 0.37350 | 12.00 |
| 6 | Frozen fusion | 0.29095 | 0.37968 | 27.20 |
| 12 | Frozen fusion | 0.29102 | 0.37991 | 33.28 |
| 24 | Dense | 0.29663 | **0.38263** | **45.45** |
| 48 | RRF | 0.29616 | 0.38296 | 69.77 |
| 90, all 100 documents | Dense pool | 0.29960 | 0.38206 | 112.34 |

At 34 documents, the paired difference versus the 100-document fallback is +0.00057 [−0.00121, +0.00285] on dev and −0.00297 [−0.00858, +0.00187] on held-out training. The dev comparison with six-pair alone is +0.00912 [+0.00134, +0.01760]. The 34-document protocol uses 66 prompts per abstention and the six-pair procedure uses 12 per query. With 263 abstentions, total estimated prompt count falls from 58,302 to 23,586, a 59.5% reduction. Prompt count follows directly from the replay policy and is distinct from measured runtime. [Original 100-document evaluation](ANCHOR_OFFICIAL_DEV_20261125.md)

## Normalization on accepted lists

Add a membership-preserving reorder to the 34-document fallback. For accepted lists, use exactly the same ten documents in all four arms and independently vary where each score source is standardized. Scores from the frozen ranker are already available on all 100 documents. The 8B judge needs 18 prompts for those ten when the reference document is in the list and 20 when the reference is outside it.

| Frozen score normalization | 8B score normalization | Held-out score | Official dev score | Added prompts averaged over all 519 queries |
|---|---|---:|---:|---:|
| Accepted ten | Accepted ten | 0.30238 | 0.38066 | 8.98 |
| Entire top 100 | Accepted ten | **0.30821** | **0.38374** | **8.98** |
| Accepted ten | Entire top 100 | 0.30208 | 0.38124 | 97.66 |
| Entire top 100 | Entire top 100 | 0.30144 | 0.38371 | 97.66 |

Among the two arms that use only the local 8B scores, the 65 held-out queries favor using the full score distribution for the frozen system. This setting improves on standardizing both scores over the accepted ten by +0.00308 [+0.00019, +0.00631] on official dev. The old system's score standard deviation can be obtained from its existing scores. Changing its normalization population changes the relative influence of the two terms in the blend, even with the same ten documents and weights.

The complete local-judge policy averages **54.43** prompts and scores **0.38374**. The earlier accepted-list policy uses full 8B normalization on both routes and scores 0.38314. The paired difference is +0.00060 [−0.00188, +0.00347]. The present experiment is a score-cache replay with fresh rankings and a prompt-count estimate. [Prior accepted-list control](ROUTED_SHORTLISTS_20261125.md) · [Scores, intervals and replayed lists](../results/analysis/budgeted_anchor_fallback_20261125.json)

## Reproduction

```bash
python scripts/replay_budgeted_anchor_fallback.py --data-root data/track1_tempo \
  --oof results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --oof-scope results/analysis/anchor_edit_scope_qwen8b_oof100_20261125.json \
  --dev-scope results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json \
  --output results/analysis/budgeted_anchor_fallback_replay.json
```
