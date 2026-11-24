# Supervised selection follow-up

The fixed six-pair system remains at 0.373503 on official dev. Earlier diagnostics found that the fixed system loses potential both before and after a pair is judged. These two CPU experiments test supervision at each point using pre-existing candidate scores and judge outputs.

## Calibrating the swap decision

On the exact same six external actions, a four-class model estimated relevance outcomes for the two documents. With qrels-positive relevance, an action helps if only the challenger is relevant and harms if only the incumbent is relevant. The decision multiplied the difference in those probabilities by the slot discount. The inputs were the same for a direct ridge control. An additional fixed policy doubled the penalty for predicted harm. All fitted coefficients came from the training folds, with a separate fit for each held-out fold and a final pooled fit for official dev.

| Policy | 969-query OOF nDCG@10 | Dev nDCG@10 | Dev helpful / harmful swaps | Dev zero queries recovered |
|---|---:|---:|---:|---:|
| Original two-order agreement | 0.283352 | 0.373503 | 33 / 10 | 9 |
| Direct expected-gain regression | 0.280441 | 0.359146 | 28 / 20 | 5 |
| Separate benefit and harm | 0.280788 | 0.367737 | 32 / 20 | 4 |
| Separate outcomes, twice the harm penalty | 0.278813 | 0.366571 | 20 / 15 | 3 |

The best new dev score is 0.00577 below the frozen policy, with a paired 95% interval [−0.01488, +0.00291]. The separate estimates improve on the matched direct regression on dev. Replacing the agreement rule still increases harmful actions. Doubling the loss penalty reduces helpful swaps as well. [Per-query results](../results/analysis/swap_risk_20261124.json)

## Improving the shortlist before judgment

A logistic relevance model ranks outside documents from the same dense top 100. It uses normalized dense and CE scores, reciprocal rank, and score disagreement. It is fitted on judged relevance in the training folds; unjudged candidates are treated as zero under the benchmark convention. Both the score-based model and the best-rank union start from the same candidate pool and remove the frozen top ten. These are candidate coverage results. Relevance labels identify zero-score or pool-reachable subgroups for evaluation.

| Evaluation | Positive in first six, rank union → supervised | Positive in first twenty, rank union → supervised |
|---|---:|---:|
| OOF, 969 queries | 208 → 256 | 416 → 439 |
| Dev, 519 queries | 109 → 115 | 207 → 208 |
| OOF pool-reachable zero subgroup, 187 queries | 49 → 62 | 111 → 125 |
| Dev pool-reachable zero subgroup, 86 queries | 29 → 30 | 59 → 60 |

The OOF top-20 coverage gain is +2.49 percentage points after weighting domains equally ([+0.51, +4.55]). On dev the corresponding gain is +0.12 points ([−2.20, +2.52]). Dev top-six hit counts rise by six while the domain-macro hit rate falls by 2.13 points. The dev base and comparator CE differ from the fold models, so this is transfer across those score distributions as well as split transfer. This coverage evidence does not justify spending a new GPU run on the same features. [Shortlist results](../results/analysis/challenger_shortlist_20261124.json)

## Decision

Keep the original competition configuration. The paired results bound what can be gained by recalibrating the saved scores alone. The next controlled experiment should add a text-based relevance signal to the candidates that the current comparison stage misses. Hold the candidate pool, six judged pairs, and maximum one replacement fixed in the system comparison. Use out-of-fold outcomes to check positive acceptance and harmful swaps before transferring its rule to official dev.

The [semantic window follow-up](SEMANTIC_JUDGE_WINDOWS_20261124.md) runs that fixed-pair comparison on f0.

## Reproduction

Run from the repository root with official qrels, the saved action caches, and the original dense runs. The shortlist experiment also needs the generated fold CE runs.

```bash
python scripts/eval_swap_risk.py --data-root data/track1_tempo --run-root runs --output results/analysis/swap_risk_20261124.json
python scripts/eval_challenger_shortlist.py --output results/analysis/challenger_shortlist_20261124.json
```
