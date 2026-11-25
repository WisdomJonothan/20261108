# Select a ranking or restore one displaced document

The previous two-excerpt policy scores 0.39488 on the 519-query official dev set. It admits 68 judged relevant documents and removes 63 relative to the pre-reranking lists. This study uses cached scores to test decisions at two levels. A query-level model chooses among three complete lists. A document-level model revises one position in the two-excerpt list. The 30-document candidate pools, Qwen3-Reranker-4B checkpoint and underlying scores are fixed. An execution that needs both evidence views calculates 60 pointwise scores per query, 30 more than the previous two-excerpt policy. [Candidate pools, evidence views and score attribution](RERANKING_HEADROOM_20261125.md)

## Calibration and official scores

Use the same 186 training-fold queries, including 65 with the exact 8B-backed starting system. Fit each model on two folds and predict the third. For the nested selection results, choose a policy using additional held-out predictions inside the two fit folds. For the official replay, choose the best fixed policy from the pooled out-of-fold predictions on the three training folds, then refit on all 186 queries. The previously inspected official dev labels enter only the reported replay metrics.

| Policy | Nested 186-query score | Nested 65-query exact reference | 519-query official dev |
|---|---:|---:|---:|
| Fixed two-excerpt list | 0.32805 | 0.32905 | 0.39488 |
| Query-level list selection | **0.32963** | **0.33647** | 0.39536 |
| Restore one displaced document | 0.32764 | 0.32633 | **0.39589** |
| List selection with repair on its two-excerpt branch | 0.32922 | — | 0.39534 |

The 13-domain macro average weights each domain equally. The nested results apply fold-specific policy choices. The pooled repair rule has a training-fold score of 0.32834 and a 65-query exact-reference score of 0.32633. The pooled query selector scores 0.32976 on its out-of-fold predictions. The official scorer independently reproduces all list scores.

## Query-level selection

Fit domain-weighted ridge regression to each reranked list's nDCG improvement over the original list. The inputs describe admission margins, agreement between the two views, how much the list changes, and the evidence score of the original first result. At inference, choose the reranked list with the larger predicted gain when that gain clears the policy threshold. Otherwise retain the original list. Controls compare two regularization strengths, three minimum-gain thresholds and policies that default to the two-excerpt list.

The pooled training choice uses regularization 40 and threshold zero. On dev, it distributes the 519 queries across the original list (72), the single-excerpt list (147) and the two-excerpt list (300). Its official improvement over the fixed two-excerpt list is **+0.00048** [−0.00574, +0.00653]. Fifty queries improve and 42 decline. Domain-macro annotated-step coverage is 64.93%. Omitting Workplace changes the descriptive gain to −0.00231. The qrels-based oracle over the three actual lists is **0.41843**. It can improve over the fixed two-excerpt list on 158 queries. [Rankings, decisions and fold audit](../results/analysis/evidence_query_router_20261125.json)

## Document-level repair

Fit a domain-weighted logistic relevance model on documents from the training-fold pools. Its inputs include the original rank, each view's raw and standardized evidence score, their agreement and membership in the two reranked lists. At inference, find a displaced document from the original top ten and compare replacing each document at ranks 2–10 in the two-excerpt list. Estimate the swap's value as the predicted relevance difference multiplied by the discount for that rank. Perform at most one positive-valued swap. Matched controls allow candidates from the single-excerpt list or require a greater predicted gain.

The pooled training choice restores from the original top ten with a zero threshold. Official dev increases by **+0.00101** [−0.00147, +0.00389]. The policy edits 136 queries, improving seven and lowering three. Counts of judged relevant documents lost relative to the original ranking fall from 63 to 56. Counts of new judged relevant documents fall from 68 to 65. The first result is preserved on every query. Domain-macro annotated-step coverage rises from **65.25% to 65.65%**. Omitting Workplace leaves a difference of −0.00002. The nested training score and the domain omission quantify the sensitivity of the gain. The composed control repairs only queries routed to the two-excerpt list. It reaches 0.39534, below the individual repair. [Model controls, per-query ranks and composition](../results/analysis/evidence_document_repair_20261125.json)

## Reproduction

Build the training and dev manifests and unpack the score archives using the [previous procedure](RERANKING_HEADROOM_20261125.md#reproduction). Keep the same manifest, data directory and cached score files for these two commands.

```bash
python scripts/replay_evidence_query_router.py --data-root data/track1_tempo \
  --training-manifest "$work/train.json" --dev-manifest "$work/dev.json" \
  --training-scores "$work/train-scores"/score-train-{temporal,dispersed}.json \
  --dev-scores "$work/temporal-scores"/*.json "$work/dispersed-scores"/*.json \
  --output "$work/query-router.json"

python scripts/replay_evidence_document_repair.py --data-root data/track1_tempo \
  --training-manifest "$work/train.json" --dev-manifest "$work/dev.json" \
  --training-scores "$work/train-scores"/score-train-{temporal,dispersed}.json \
  --dev-scores "$work/temporal-scores"/*.json "$work/dispersed-scores"/*.json \
  --router-report "$work/query-router.json" --output "$work/document-repair.json"
```
