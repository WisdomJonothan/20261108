# Fixed-budget candidate shortlists and accepted-list protection

## Restricting edits after six-pair acceptance

Use the existing 8B scores and the six-pair starting list. When six-pair accepts a replacement, keep the ten documents it selected and allow their relative ordering to change. When it abstains, use the existing unrestricted fused ranking. Reuse all candidate pools, query and passage inputs, and fusion scores. The rule applies to the 65 held-out queries with train-fold upstream scores and the 519 official dev queries.

| Policy | Held-out 65 | Official dev 519 |
|---|---:|---:|
| Six-pair | 0.26976 | 0.37350 |
| Unrestricted 8B on every query | 0.31986 | 0.37959 |
| Unrestricted 8B on abstentions | 0.29960 | 0.38206 |
| Unrestricted 8B on abstentions, reorder accepted top tens | 0.30441 | 0.38314 |

Relative to the previous abstention fallback, the same-ten edit for accepted queries contributes +0.00481 on held-out queries and +0.00109 on official dev, with paired within-domain intervals [−0.00559, +0.01585] and [−0.00678, +0.00842]. Relative to unrestricted 8B on all queries, the combined rule changes the scores by −0.01544 and +0.00355, respectively. On official dev, restricting edits after acceptance avoids the decline from unrestricted ranking in that subset and retains the gains on abstentions. The held-out comparison favors unrestricted ranking in both groups. This is a secondary replay of fixed scores. [Both splits, groups, and query-level rankings](../results/analysis/accepted_membership_20261125.json)

## Six challengers per source

Start each query with the archived 0.38206 abstention-fallback run. Each arm uses six distinct challengers. The sources are mxbai, E5 and BGE dense runs. For the first three arms, select six by reciprocal rank fusion with offset 60 over the three top-300 runs. The fourth arm uses mxbai rank order directly.

| Candidate source | Queries with a judged relevant candidate | Zero-score queries with a judged relevant candidate | Rank-ten oracle gain |
|---|---:|---:|---:|
| mxbai top 100 outside the starting ten | 72 | 15 | +0.01195 |
| mxbai ranks 101–300 | 20 | 4 | +0.00211 |
| E5 or BGE top 300 outside mxbai top 300 | 8 | 3 | +0.00060 |
| mxbai ranks 101–106 | 14 | 5 | +0.00467 |

The oracle retains the first nine positions and makes the best possible choice at position ten using qrels. On these shortlists, the gains available from the first 100 exceed the gains from each expansion arm. The ranking budget explains part of the gap with the earlier depth-300 oracle, which had access to 200 new candidates and could reorder the entire list. [Label-based audit for all 519 queries](../results/analysis/routed_shortlists_oracle_20261125.json) · [Earlier full-depth oracle](ADAPTIVE_CANDIDATE_DEPTH_20261125.md)

## 26-query inference check

Select two queries per domain using a fixed seed and use Qwen3-4B-Instruct-2507 in BF16 to score independent passages. Use the native template, the same lexical 112-word window selector as the 8B experiment, and shorten individual words of 2,048 or more characters to 256 characters. Rank by the raw logit difference between yes and no. A challenger replaces position ten only if its score strictly exceeds the incumbent's. The equal per-arm budget is six challengers and the incumbent. A secondary comparison reranks the resulting ten. Score 863 unique query-document prompts across all arms in 65 seconds, including preparation.

| Arm | Rank-ten replacement | Sort after the same replacement |
|---|---:|---:|
| Starting list | 0.33437 | 0.30691 |
| mxbai top 100 outside the starting ten | 0.34393 | 0.31768 |
| mxbai ranks 101–300 | 0.33437 | 0.30444 |
| E5 or BGE outside mxbai top 300 | 0.33437 | 0.30496 |
| mxbai ranks 101–106 | 0.33437 | 0.30461 |

Two previously nonzero queries improve under the top-100 intervention and zero decline. The three other fixed-slot arms produce equal nDCG on all 26 queries. The full-dev comparison scores only the position-ten incumbent and four challenger lists. [Preflight output and scores](../results/analysis/routed_shortlists_preflight_20261125.json)

## Reproduction

```bash
python scripts/replay_accepted_membership.py --oof results/analysis/anchor_edit_scope_qwen8b_oof100_20261125.json --dev results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json --output results/analysis/accepted_membership_replay.json

python scripts/eval_routed_shortlists.py --reference results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json --routes results/analysis/adaptive_candidate_depth_dev_20261125.json --write-manifest /tmp/reteco-shortlists.json --output results/analysis/routed_shortlists_oracle_replay.json

python scripts/eval_routed_shortlists.py --manifest /tmp/reteco-shortlists.json --model models/Qwen3-4B-Instruct-2507 --boundary-only --checkpoint /tmp/reteco-shortlists-boundary --output results/analysis/routed_shortlists_official_dev_replay.json
```
