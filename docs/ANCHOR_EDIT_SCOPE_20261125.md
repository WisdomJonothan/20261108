# Candidate admission and reordering with fixed relevance scores

The [edit-budget oracle](SWAP_EVIDENCE_SPANS_20261125.md) compares an ideal fixed-position swap with an ideal reorder. This replay measures what the existing score actually does at the same candidate depth and inference budget. The shared-reference comparisons follow the [previous experiment](ANCHOR_RERANKER_20261124.md), with a connection to [RefRank](https://arxiv.org/abs/2506.11452). The fixed-score intervention separates the effect of admitting candidates from the effect of changing their positions in the final fused ranking. [Rerank Before You Reason](https://aclanthology.org/2026.findings-acl.1289/) examines ranking model scale and depth in BrowseComp-Plus. [Active Learners as Efficient PRP Rerankers](https://aclanthology.org/2026.findings-emnlp.1216/) studies how to allocate pairwise judgments to identify the top-k. These provide relevant depth and call-budget controls for future benchmark transfer.

Each row uses one fixed set of documents and one score per document. The score is the previous 70:30 blend of standardized frozen dense-plus-CE scores and shared-reference log odds against frozen rank five. The best-scoring outsider enters only if it outranks the weakest-scoring incumbent. The two one-admission arms keep exactly the same ten documents. Their difference is whether the newcomer takes the vacated slot or all ten are sorted by the same scores. The membership-preserving reorder is an additional control. The unrestricted arm takes the ten highest-scoring documents in the pool. Qrels enter the metric and oracle calculations after selection.

## Matched replay with Qwen3-4B-Instruct-2507

| Sample and starting list | Start | Reorder same ten | Admit one in place | Admit same one and reorder | Unrestricted |
|---|---:|---:|---:|---:|---:|
| 65 held-out training-fold queries, dense top 100, frozen fusion | 0.27369 | 0.27965 | 0.27905 | 0.28733 | 0.28703 |
| Same 65, six-pair competition output | 0.26976 | 0.27778 | 0.27723 | 0.28642 | 0.28703 |
| 519 official dev queries, protected pool of 20, frozen fusion | 0.35810 | 0.36341 | 0.36353 | 0.36823 | 0.36817 |
| Same 519, six-pair output with edits on abstentions only | 0.37350 | 0.37262 | 0.37549 | 0.37453 | 0.37411 |

On the 65 training-fold queries starting from the competition list, sorting the identical admission-selected membership adds +0.00918 (paired within-domain bootstrap 95% interval [−0.00353, +0.02179], domain bootstrap [−0.01195, +0.03075]). Leaving out any one domain gives +0.00319 to +0.01502. The effect is +0.01581 on the first 26 and +0.00477 on the disjoint 39. Unrestricted admission then adds +0.00061 [−0.00809, +0.01030] even though it changes more than one document for 53 of 65 queries. Relative to the competition start, unrestricted admission gains +0.02360 from relevant arrivals and loses −0.01211 from relevant departures, with +0.00578 from retained documents moving in rank. These quantities sum exactly to the macro gain.

On dev, reordering the same one-admission membership adds +0.00471 [−0.00485, +0.01388] from the frozen start. Broader admission changes the point estimate by −0.00006 [−0.00135, +0.00128]. These fixed-score comparisons put the larger positive point estimate on position changes within the selected ten.

For edits after the competition stage, 176 of the 256 accepted competition arrivals fall outside the cached protected 20-document pool. The all-query comparison therefore keeps accepted six-pair edits and operates only on the 263 abstentions. It requires 38 extra prompts on those queries, or 31.26 total prompts per query on average. Here the single fixed-slot admission is the highest point estimate at 0.37549. The gain is +0.00199 [−0.00183, +0.00616]. It recovers seven of the 130 competition zeroes. The abstention subset produces a different ordering tradeoff from the frozen-start replay. The six-pair competition configuration scores 0.37350 with 12 prompts per query.

## How much better could selection over these lists be?

For each query, the proposal oracle chooses the highest official score among the five lists above, including the starting list. On official dev it reaches 0.38488 from the frozen start and 0.38467 when editing only on competition abstentions. On the 65 training-fold queries starting from the competition result and using the whole dense top 100, the proposal oracle reaches 0.30879. These are ceilings for choosing among outputs actually proposed by these scores. The prior 0.69850 official-dev ceiling uses a larger dense top-100 pool and ideal doc-level judgments, and answers a different question.

[Held-out per-query results](../results/analysis/anchor_edit_scope_oof100_20261125.json) · [Official-dev per-query results](../results/analysis/anchor_edit_scope_dev20_20261125.json)

## Matched Qwen3-8B checkpoint comparison

The larger checkpoint uses the original full query, lexical excerpts, rank-five anchor, Direct prompt, both presentation orders, 198 calls per query, and the frozen 70:30 blend. Its native chat template uses non-thinking mode for the one-token answer. Query, fold, anchor and all 100 document IDs are matched to the previously scored Qwen3-4B-Instruct-2507 cohorts. The first 26 and the independent 39 have disjoint query IDs. [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)

| Held-out cohort | 4B unrestricted | 8B unrestricted | Paired difference |
|---|---:|---:|---:|
| First 26 | 0.28512 | 0.33647 | +0.05135 |
| Independent 39 | 0.28830 | 0.30878 | +0.02048 |
| Combined 65 | 0.28703 | 0.31986 | +0.03283 |

The pooled paired within-domain 95% interval is [+0.01576, +0.05030]. The domain-resampling interval is [+0.00006, +0.06684]. Nine of the thirteen domains improve and four decline. The independent 39-query interval is [+0.00247, +0.03791] within the evaluated domains and [−0.01652, +0.05722] in the domain-resampling check. The two cohorts use five total queries per domain.

| Output from the six-pair start, dense top 100 | 4B | 8B | 8B minus 4B |
|---|---:|---:|---:|
| Reorder the same ten | 0.27778 | 0.29381 | +0.01603 |
| Admit one in place | 0.27723 | 0.27973 | +0.00250 |
| Admit the same one and reorder | 0.28642 | 0.30146 | +0.01504 |
| Unrestricted | 0.28703 | 0.31986 | +0.03283 |

Going from one admission with free reordering to unrestricted admission adds +0.01840 for 8B and +0.00061 for 4B. The difference of +0.01779 has a paired within-domain interval [+0.00287, +0.03331] and a domain-resampling interval [+0.00225, +0.03510]. The direction repeats across the 26- and 39-query cohorts. The 8B unrestricted ranking's gain over the six-pair start consists of +0.03260 from relevant arrivals, −0.00446 from departures and +0.02195 from relevant documents changing position within the ten. It recovers five of the 22 zeroes in that starting list.

Holding the membership of the candidate pool to twenty in a cache replay gives 0.29256 for 4B and 0.30689 for 8B. For 8B, using all 100 candidates adds +0.01297 [paired within-domain interval +0.00117, +0.02572] over its identical 20-document protocol. This direction repeats with +0.02664 on the first 26 and +0.00386 on the independent 39. The direct 8B runs used 12,870 prompts, 30.8 minutes of scoring and up to 17,249 MiB of allocated GPU memory. The evidence favors testing the fixed 8B 70:30 blend at depth 100 on all 519 official dev queries, with the six-pair output as the reference and the protected 20-document replay as the depth control.

[8B score cache, first 26](../results/analysis/anchor_reranker_qwen8b_sample26_20261125.json) · [8B score cache, independent 39](../results/analysis/anchor_reranker_qwen8b_replication39_20261125.json) · [Paired 100-document replay](../results/analysis/anchor_edit_scope_qwen8b_oof100_20261125.json) · [Paired 20-document replay](../results/analysis/anchor_edit_scope_qwen8b_oof20_20261125.json)


In the first cohort, a query about the time scope of a presidential pardon (law, `106290_825`) illustrates the position effect. The competition run places the judged relevant example of a fixed-period pardon fifth. The 4B unrestricted blend moves it to fourth and the 8B unrestricted blend moves it to first. Under the 8B fixed-position edit it stays fifth.


## Reproduction

The compact frozen-score caches include the ordered candidate IDs and original scores used to produce the published anchor rankings. They allow the replay to run with the official qrels without reconstructing the held-out scoring runs.

```bash
python scripts/replay_anchor_edit_scope.py --data-root data/track1_tempo --pool-scores results/analysis/anchor_edit_scores_oof100_20261125.json --inputs results/analysis/anchor_reranker_oof_sample26_20261124.json results/analysis/anchor_reranker_oof_replication39_20261124.json --reference results/analysis/position_oracle_oof_20261124.json --starts frozen competition abstention --output results/analysis/anchor_edit_scope_oof100_replay.json

python scripts/replay_anchor_edit_scope.py --data-root data/track1_tempo --pool-scores results/analysis/anchor_edit_scores_dev20_20261125.json --inputs results/analysis/anchor_cap20_official_dev_20261124.json --reference results/analysis/position_oracle_dev_20261124.json --starts frozen abstention --output results/analysis/anchor_edit_scope_dev20_replay.json
```


Run the matched 8B checkpoint with the original fold runs and corpus available. The matches below verify the exact query and fold membership before inference.

```bash
python scripts/eval_anchor_reranker.py --model models/Qwen3-8B --disable-thinking --queries-per-domain 2 --pool-depth 100 --match-report results/analysis/anchor_reranker_oof_sample26_20261124.json --output results/analysis/anchor_reranker_qwen8b_sample26_20261125.json

python scripts/eval_anchor_reranker.py --model models/Qwen3-8B --disable-thinking --queries-per-domain 3 --pool-depth 100 --exclude-report results/analysis/anchor_reranker_oof_sample26_20261124.json --match-report results/analysis/anchor_reranker_oof_replication39_20261124.json --output results/analysis/anchor_reranker_qwen8b_replication39_20261125.json
```

Use the already-scored documents for the matched model comparison. The depth-20 replays retain frozen top-ten membership in the pool, add the first ten dense outsiders and recompute both z-scores within those 20 documents.

```bash
python scripts/replay_anchor_edit_scope.py --data-root data/track1_tempo --pool-scores results/analysis/anchor_edit_scores_oof100_20261125.json --inputs results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json --reference results/analysis/position_oracle_oof_20261124.json --compare results/analysis/anchor_edit_scope_oof100_20261125.json --starts frozen competition abstention --output results/analysis/anchor_edit_scope_qwen8b_oof100_replay.json

python scripts/replay_anchor_edit_scope.py --data-root data/track1_tempo --pool-scores results/analysis/anchor_edit_scores_oof100_20261125.json --inputs results/analysis/anchor_reranker_oof_sample26_20261124.json results/analysis/anchor_reranker_oof_replication39_20261124.json --reference results/analysis/position_oracle_oof_20261124.json --protected-depth 20 --starts frozen abstention --output results/analysis/anchor_edit_scope_oof20_replay.json

python scripts/replay_anchor_edit_scope.py --data-root data/track1_tempo --pool-scores results/analysis/anchor_edit_scores_oof100_20261125.json --inputs results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json --reference results/analysis/position_oracle_oof_20261124.json --protected-depth 20 --compare results/analysis/anchor_edit_scope_oof20_20261125.json --starts frozen abstention --output results/analysis/anchor_edit_scope_qwen8b_oof20_replay.json
```


## Official-dev protocol prepared for the next run

Use the frozen 8B checkpoint and the existing 70:30 blend on the entire dense top 100 of the 519 official dev queries. The protected depth-20 comparison can be replayed from the same scores. The primary comparison is domain-macro nDCG@10 against the six-pair competition run on the same queries. The 100-document run requires 102,762 prompts, approximately 4.1 hours at the measured 3090 throughput.

```bash
python scripts/eval_anchor_reranker.py --model models/Qwen3-8B --disable-thinking --split dev --pool-depth 100 --output results/analysis/anchor_reranker_qwen8b_dev100.json
```
