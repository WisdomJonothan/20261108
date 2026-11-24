# Confirming changes to the first three ranks

The [shared-reference study](ANCHOR_RERANKER_20261124.md) found that its fallback increased judged relevance in the top ten while losing discounted gain at the first three ranks on official dev. Locking those ranks lowered performance in both of the earlier held-out samples. We test a selective rule using the same Qwen3-4B-Instruct judge. The rule requires the challenger to win under both document presentation orders, with a mean challenger probability above 0.5.

## Existing CE exchange proposals

On the 969 held-out training-fold queries, keep accepted six-pair external decisions. When the external policy abstains, offer only cached internal exchange proposals whose higher slot was originally ranked first, second, or third. Compare the highest mean challenger probability above 0.5 with and without the agreement requirement. Both methods consider the same actions, use both presentations, and take at most one exchange.

| Policy | Domain-macro nDCG@10 | Helpful / harmful extra exchanges | Mean prompts |
|---|---:|---:|---:|
| Six-pair external | 0.28335 | — | 12.00 |
| CE-proposed exchange, mean probability alone | 0.28244 | 39 / 34 | 16.22 |
| CE-proposed exchange, two-order confirmation | 0.28457 | 32 / 25 | 16.22 |

Agreement removes nine harmful and seven helpful exchanges in aggregate. Its paired gain over mean probability alone is +0.00212 (within-domain bootstrap 95% interval [−0.00298, +0.00965]). Its gain over six-pair external is +0.00122 [−0.00265, +0.00507]. Ideal selection among the cached head proposals would add +0.02200, or +0.00717 when restricted to proposals that pass confirmation. [Replay results](../results/analysis/head_agreement_oof_20261124.json)

## Shared-reference proposals

On a six-pair abstention, start with the original first three documents and fill ranks 4–10 from the 20-document shared-reference ranking. Find the first slot among positions 1–3 at which the shared-reference ranking differs. That proposed document is already in the protected list. A single exchange makes it occupy the proposed rank. Use the same Direct prompt, query, 112-word lexical excerpts and two document orders as the existing judge to confirm this exchange. For a matched control, apply the same exchange directly. Accepted replacements from the original six-pair judge remain in place throughout.

In the 65 previously sampled training-fold queries, 23 head proposals require 46 fresh prompts. Eleven pass confirmation. All three score-increasing exchanges pass; the twelve rejected proposals leave nDCG unchanged. With and without confirmation, the single exchange scores 0.27711, against 0.27436 for fixed protection and 0.27793 for the unprotected reference ranking. These reused samples helped motivate the experiment. [Fresh training-fold judgments](../results/analysis/anchor_head_confirmation_oof_20261124.json)

### Official dev

The same rule produces head proposals for 186 of the 263 external-six abstentions. The direct judge confirms 124, using 372 fresh prompts total.

| Policy | Domain-macro nDCG@10 | Mean prompts per query |
|---|---:|---:|
| Six-pair external | 0.37350 | 12.00 |
| Unprotected shared-reference fallback | 0.37411 | 31.26 |
| Fixed protection of original first three | 0.37554 | 31.26 |
| Take the first head proposal directly | 0.37407 | 31.26 |
| Take the first head proposal after confirmation | 0.37526 | 31.97 |

On the same 186 proposed exchanges, confirmation retains 19 of 25 helpful exchanges and 18 of 23 harmful exchanges. It improves the identical ungated proposal by +0.00119 [−0.00398, +0.00778]. Relative to fixed protection, confirmation changes nDCG by −0.00028 [−0.00388, +0.00312]. Perfect decisions on the same first proposed exchange would add at most +0.00501 to fixed protection. This ceiling uses qrels only for analysis.

Fixed protection gains +0.00204 over six-pair on dev [−0.00206, +0.00656] and loses to the unprotected fallback on both of the training-fold samples. These dev comparisons are exploratory. The competition configuration remains six-pair external at 12 prompts. The matched cases indicate that two-order consistency reduces activity, with similar retention rates on helpful and harmful proposals. Further rounds of the same check have limited headroom, and an in-list swap leaves a zero-score top ten at zero. [Official-dev judgments and paired comparisons](../results/analysis/anchor_head_confirmation_dev_20261124.json)

## Reproduction

Run from the repository root with the official corpus and the cached runs referenced below.

```bash
python scripts/replay_head_agreement.py --input results/analysis/position_repair_oof_20261124.json --output results/analysis/head_agreement_oof_20261124.json
python scripts/eval_anchor_head_confirmation.py --model models/Qwen3-4B-Instruct-2507 --reference results/analysis/position_oracle_oof_20261124.json --input results/analysis/anchor_reranker_oof_sample26_20261124.json results/analysis/anchor_reranker_oof_replication39_20261124.json --cap results/analysis/anchor_cap20_oof_sample26_20261124.json results/analysis/anchor_cap20_oof_replication39_20261124.json --output results/analysis/anchor_head_confirmation_oof_20261124.json
python scripts/eval_anchor_head_confirmation.py --model models/Qwen3-4B-Instruct-2507 --reference results/analysis/position_oracle_dev_20261124.json --input results/analysis/anchor_cap20_official_dev_20261124.json --output results/analysis/anchor_head_confirmation_dev_20261124.json
```
