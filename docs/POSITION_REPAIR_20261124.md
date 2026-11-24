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

## Qwen evaluation

The scores for the position-exchange and mixed policies are pending. The frozen competition configuration remains the reference. We will compare the two new arms to the same six-pair external arm, report the actual changes and costs, and keep the fold and dev results separate.

Run from the project root with the official data, fold CE TREC runs, and the frozen cache described in [the replacement report](../results/replacement_v2/REPORT.md). The training CE runs are generated artifacts.

```bash
python scripts/eval_position_repair.py --folds f0 f1 f2 --output results/analysis/position_oracle_oof_20261124.json
python scripts/eval_position_repair.py --folds dev --output results/analysis/position_oracle_dev_20261124.json
python scripts/eval_position_rank_features.py --output results/analysis/position_rank_features_20261124.json
python scripts/eval_position_repair.py --model models/Qwen3-4B-Instruct-2507 --folds f0 f1 f2 --output results/analysis/position_repair_oof_20261124.json
python scripts/eval_position_repair.py --model models/Qwen3-4B-Instruct-2507 --folds dev --output results/analysis/position_repair_dev_20261124.json
```
