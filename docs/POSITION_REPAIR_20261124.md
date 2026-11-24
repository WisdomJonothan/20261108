# Fixed-budget position repair · 2026-11-24

The competition configuration scores 0.373503 on official dev and improves the pooled training folds by 0.004483. It uses six comparisons to consider a new document for one of the ten returned positions. A relevant document that is already in the ten can also gain nDCG when it moves upward. We test whether spending the same number of comparisons on both kinds of change improves the final list.

The comparison model, two presentation orders, and one-action limit come from the existing system. The external arm replays the original six head-first judgments. The internal arm chooses six pairs from the ten returned documents by multiplying the alternative CE's standardized score difference by the difference in rank discount. Promoting a lower document exchanges its position with the higher document. The mixed arm takes three actions of each kind. All three arms take the highest-confidence eligible action. Eligibility requires the challenger to win in both presentation orders and to have average probability above 0.5. The comparison count is held at six pairs. Input token counts can differ with passage lengths. Qrels are used for scoring and the oracle calculation.

Pairwise LLM prompting and budget-aware reranking have prior work, including [Pairwise Ranking Prompting](https://aclanthology.org/2024.findings-naacl.97/) and [EcoRank](https://arxiv.org/abs/2402.10866). The RETECO question here is specific to a mostly fixed fusion ranking: whether the same judgment budget is better spent adding new evidence or promoting evidence that fusion has already found.

| Evaluation | Start | External-six oracle | Internal-six oracle | Mixed three-plus-three oracle |
|---|---:|---:|---:|---:|
| Training out-of-fold, 969 queries | 0.278869 | 0.311088 | 0.325102 | 0.339651 |
| Official dev, 519 queries | 0.358099 | 0.401902 | 0.410366 | 0.423679 |

In the training folds, 219 queries have an improvable position exchange but no beneficial external action in the six-action baseline. On dev this holds for 142 queries. Conversely, on dev, 50 have only an improvable external action. An internal exchange leaves a zero-support list at zero. The six external actions reach a beneficial candidate on 23 of the 138 zero-score dev queries. The mixed oracle is higher in 11 of 13 domains on each split, with different domains losing. These are ceilings over the fixed actions, calculated with qrels.

## CPU control

A pairwise logistic rank model uses only reciprocal log ranks from the frozen ranking, dense ranking, and CE ranking. It is fitted on two available training folds and evaluated on the third, and the final training-fold fit is applied to dev. The fitted rank features are unchanged after looking at dev.

| Evaluation | One exchange | Full reorder of the same ten |
|---|---:|---:|
| Training out-of-fold, delta versus start | +0.00316 | +0.00273 |
| Official dev, delta versus start | −0.00170 | +0.00116 |

On dev, the full reorder interval is [−0.00519, +0.00720] and the one-exchange interval is [−0.00778, +0.00377]. The rank features provide a useful cost-matched action-pool control. They show no confirmed dev gain. The CE in the training fold is the fold reranker, while dev uses the existing full-question CE comparator. Feature transfer across those settings is part of the measured test.

We also applied the CPU exchange only when the six-pair external judge declined all actions. This adds 0.00256 on the training folds (95% interval [−0.00022, +0.00566]) and loses 0.00297 on dev ([−0.00792, +0.00117]), moving 0.37350 to 0.37053. The fallback is an exploratory control, so the current competition configuration is retained. [Replay result](../results/analysis/position_fallback_20261124.json)

## Starting-ranker control

We averaged the three full-query CE seeds after query-level standardization on the same top 100 and standardized the mean before fusion. Transferring the seed-17 internally selected CE weight of 0.7 yields 0.36242 on dev. Equal fusion of the same scores yields 0.34598. The corresponding three-seed title fusion gives 0.34953, and equal title/full view fusion gives 0.35720. These outcomes are exploratory and use cached scores. The ensembles would require extra CE passes at inference.

The original six-pair judge selected 256 replacement actions. Only 26 still apply directly to the 0.7 ensemble, because the old incumbent must still be present and the challenger must still be absent. The original incumbent was absent in 147 cases, and the challenger was already present in 173, with overlap. Replaying the 26 eligible actions yields 0.36448, versus 0.37350 for the original configuration. This experiment measures portability of a fixed set of decisions. A fresh comparison on the new base would rebuild the actions. [Ensemble and transfer results](../results/analysis/ce_seed_ensemble_transfer_20261124.json)

## Qwen evaluation

Training-fold inference is complete.

| Six-pair policy | Fold nDCG@10 | Helpful / harmful | Delta vs external-six | 95% paired interval |
|---|---:|---:|---:|---|
| External only | 0.283352 | 45 / 25 | — | — |
| Internal only | 0.286591 | 90 / 43 | +0.003239 | [−0.00406, +0.01164] |
| Three external + three internal | 0.285578 | 73 / 52 | +0.002226 | [−0.00522, +0.01061] |

The Qwen internal judgments averaged about 11 prompts/s over the three folds. The paired intervals resample queries within each domain for these fixed runs.

| Six-pair policy | Official dev nDCG@10 | Helpful / harmful | Delta vs external-six | 95% paired interval |
|---|---:|---:|---:|---|
| External only | 0.373503 | 33 / 10 | — | — |
| Internal only | 0.367608 | 60 / 22 | −0.005895 | [−0.01711, +0.00449] |
| Three external + three internal | 0.365439 | 55 / 22 | −0.008064 | [−0.01767, +0.00032] |

The new arms have higher action-level oracle scores and more helpful changes, alongside roughly twice as many harmful changes. Their fold point estimates exceed the external judge, and their dev point estimates are lower. The reference competition configuration stays at 0.3735. [Fold results](../results/analysis/position_repair_oof_20261124.json) · [Dev results](../results/analysis/position_repair_dev_20261124.json)

An exploratory cache replay replaced maximum confidence with a discount-weighted priority on the same accepted pairs. On the folds, mixed falls by 0.00055 and internal-only rises by 0.00034 relative to its confidence control. Of the 270 queries with a beneficial mixed action in the judged set, 152 have no beneficial action accepted, while 39 have one accepted but the discount rule picks a nonbeneficial action. This directs attention to acceptance errors and false positives at retrieved positions. [Rank-discount replay](../results/analysis/position_discount_oof_20261124.json)

On dev the discount priority also falls relative to confidence ranking in all three arms. [Dev discount replay](../results/analysis/position_discount_dev_20261124.json)

The oracle ladder separates the two remaining errors. On dev the mixed policy offers +0.06558 under perfect choice, +0.02390 under perfect choice restricted to pairs accepted by Qwen, and +0.00734 in the actual run. The corresponding gains for the external policy are +0.04380, +0.02220, and +0.01540. Acceptance removes most of the mixed policy's extra opportunity. Within the accepted set, the mixed policy also loses more value in its final selection. [Headroom audit](../results/analysis/position_headroom_20261124.json)

The current 0.3735 system recovers nine zero-score queries from the reference ranking and creates one new zero, leaving 130. Of those, 78 have a judged positive in dense top 100, 42 have their first judged positive at ranks 101–1000, and 10 have none in top 1000. The next score-improvement target is the judged-positive selection problem in the first group. Candidate-pool expansion addresses the second group. Gold labels define these diagnostic buckets. [Competition zero-query audit](../results/analysis/competition_zeroes_20261124.json)

Tracing the 78 pool-reachable zeroes gives 51 with a judged positive in the 20 outside candidates, 25 with a positive challenger in the 18-action shortlist, 14 with a beneficial action in the six judged comparisons, and 3 with a beneficial accepted action. These counts describe the fixed system. They show both shortlist losses and judge losses. [Action trace](../results/analysis/competition_zero_action_trace_20261124.json)

Giving distinct challengers priority within the cached 18 actions increases mean distinct challengers per six-pair budget from 4.13 to 5.79 on dev, and raises the fixed-action oracle from 0.40190 to 0.41150. Realized dev nDCG is 0.36940, a change of −0.00410 (95% paired interval [−0.01171, +0.00164]). Out of fold the change is +0.00011, with interval [−0.00180, +0.00190]. The extra candidate exposure gives no reliable final gain. [Diverse-challenger replay](../results/analysis/distinct_challengers_20261124.json)

## Fresh actions on the alternate ranking

The fresh external run uses the fixed 0.7 ensemble and the original action construction. It ranks the 200 possible actions by the inexpensive score gap, keeps 18, then judges the first six in head-first order. Running this construction on the original base reproduces the exact action lists on 519/519 queries.

On the ensemble, 6,228 prompts took 663 seconds of judge scoring. The six-pair policy improves 0.36242 to 0.36583 (+0.00341, 95% interval [−0.00056, +0.00841]), with 13 helpful and 13 harmful replacements. On the same dev queries the ensemble pipeline is 0.00767 below the original 0.37350 configuration, with paired interval [−0.02148, +0.00582]. Its incremental judge gain is 0.01199 lower ([−0.02191, −0.00354]). The better standalone ranking had less measured benefit from the same remedy even after it received fresh actions. [Matched comparison](../results/analysis/ce_seed_ensemble_fresh_comparison_20261124.json)

The [supervised selection follow-up](SWAP_RISK_FOLLOWUP_20261124.md) tests separate benefit and harm probabilities and a train-fitted candidate shortlist. Both leave the original competition entry in place.

Run from the project root with the official data, fold CE TREC runs, and the frozen cache described in [the replacement report](../results/replacement_v2/REPORT.md). The training CE runs are generated artifacts.

```bash
python scripts/eval_position_repair.py --folds f0 f1 f2 --output results/analysis/position_oracle_oof_20261124.json
python scripts/eval_position_repair.py --folds dev --output results/analysis/position_oracle_dev_20261124.json
python scripts/eval_position_rank_features.py --output results/analysis/position_rank_features_20261124.json
python scripts/replay_position_fallback.py
python scripts/eval_ce_seed_ensemble.py --external-replay results/analysis/position_oracle_dev_20261124.json --output results/analysis/ce_seed_ensemble_transfer_20261124.json
python scripts/eval_position_repair.py --model models/Qwen3-4B-Instruct-2507 --folds f0 f1 f2 --output results/analysis/position_repair_oof_20261124.json
python scripts/eval_position_repair.py --model models/Qwen3-4B-Instruct-2507 --folds dev --output results/analysis/position_repair_dev_20261124.json
python scripts/audit_position_headroom.py
python scripts/audit_competition_zeroes.py --output results/analysis/competition_zeroes_20261124.json
python scripts/audit_competition_zeroes.py --trace-actions --output results/analysis/competition_zero_action_trace_20261124.json
python scripts/replay_distinct_challengers.py --output results/analysis/distinct_challengers_20261124.json
python scripts/eval_ce_seed_ensemble.py --export-runs runs --output results/analysis/ce_seed_ensemble_base_run_20261124.json
python scripts/expand_actions3.py --model models/Qwen3-4B-Instruct-2507 --start-tag full3ce70_20261124 --order head_first --max-actions 18 --judge-limit 6 --budgets 6 --out results/analysis/ce_seed_ensemble_fresh_replacement_dev_20261124.json
python scripts/compare_ensemble_replacement.py
```
