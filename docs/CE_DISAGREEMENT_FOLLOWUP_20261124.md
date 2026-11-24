# CE disagreement and fixed-pool retrieval follow-up

The current reference remains 0.37350 on official dev. The [position analysis](POSITION_REPAIR_20261124.md) identified missed candidates and incorrect decisions among the remaining zero-score queries. This follow-up tests whether a comparison budget reserved for ranker disagreement gives more useful actions, and whether additional retrieval sources can expose missing positives at a fixed pool depth.

## Reserve three comparisons for CE disagreement

Keep the first three head-first actions from the reference policy. Offer up to three distinct outsiders that are in independent CE top 20 and absent from dense top 20. Pair each with the document ranked lowest by CE among the first three incumbents, provided CE prefers the newcomer. Fill unused slots from the original shortlist. The same Qwen judge and both-order agreement rule still choose at most one swap. Qrels score actions and do not determine the candidate list. Of 969 held-out training-fold queries, 507 receive such an action. Existing scores let us reuse unchanged pairs and score 799 fresh pairs (1,598 prompts, 151 seconds of judging).

| Method | OOF nDCG@10 | Helpful / harmful swaps |
|---|---:|---:|
| Six original actions | 0.28335 | 45 / 25 |
| Three original, up to three CE disagreement actions | 0.27653 | 48 / 28 |
| Disagreement only if none of the first three actions passes | 0.27709 | 46 / 30 |

The oracle rises from 0.31109 to 0.31360. Queries with a beneficial action rise from 147 to 156. The number of zero-score queries with a beneficial action stays at 38. The paired actual nDCG change is −0.00682 (95% interval [−0.01962, +0.00124]). Iota accounts for much of the macro decline (−0.07821 across six fold queries). The query-weighted change is −0.00035. No domain-specific exception was fitted, and the rule was not extended to official dev.

[Oracle](../results/analysis/ce_disagreement_oracle_oof_20261124.json) · [Judge results](../results/analysis/ce_disagreement_judge_oof_20261124.json) · [Adaptive replay](../results/analysis/ce_disagreement_gate_oof_20261124.json)

## Complementary retrieval at the same 100 documents

Protect the actual competition top ten. Fill to 80 with mxbai results and then to 100 by alternating the BGE and E5 retrieval runs, dropping duplicates. The candidate rule does not read qrels. On official dev, pool recall changes from 0.69947 to 0.70536 and the ideal nDCG from 0.74462 to 0.75155. The paired oracle gain is +0.00693 (95% interval [−0.00635, +0.02520]). This gives five previously empty pools a relevant document but removes the only judged positive from three other pools. All eight are zeroes of the current output. This result is an exploratory retrieval ceiling, without judge or reranker scoring.

[Pool audit](../results/analysis/pool_complement_20261124.json)

## Decision

Keep the reference competition configuration. The added disagreement reaches a few more beneficial pairs, but the judge does not convert the extra reach into a gain. The pool-100 union provides only modest additional supervised coverage. This round supports testing stronger candidate relevance scores under a matched budget before further widening the six-pair rule.

```bash
python scripts/eval_ce_disagreement_actions.py --folds f0 f1 f2 --output results/analysis/ce_disagreement_oracle_oof_20261124.json
python scripts/eval_ce_disagreement_actions.py --folds f0 f1 f2 --model models/Qwen3-4B-Instruct-2507 --output results/analysis/ce_disagreement_judge_oof_20261124.json
python scripts/replay_ce_disagreement_gate.py --input results/analysis/ce_disagreement_judge_oof_20261124.json --output results/analysis/ce_disagreement_gate_oof_20261124.json
python scripts/audit_pool_complement.py --data-root data/track1_tempo --output results/analysis/pool_complement_20261124.json
```
