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

Select two queries per domain using the seed shortlist-preflight-{domain} on IDs sorted within each domain and use Qwen3-4B-Instruct-2507 in BF16 to score independent passages. Use the native template, the same lexical 112-word window selector as the 8B experiment, and shorten individual words of 2,048 or more characters to 256 characters. Rank by the raw logit difference between yes and no. A challenger replaces position ten only if its score strictly exceeds the incumbent's. The equal per-arm budget is six challengers and the incumbent. A secondary comparison reranks the resulting ten. Score 863 unique query-document prompts across all arms in 65 seconds, including preparation.

| Arm | Rank-ten replacement | Sort after the same replacement |
|---|---:|---:|
| Starting list | 0.33437 | 0.30691 |
| mxbai top 100 outside the starting ten | 0.34393 | 0.31768 |
| mxbai ranks 101–300 | 0.33437 | 0.30444 |
| E5 or BGE outside mxbai top 300 | 0.33437 | 0.30496 |
| mxbai ranks 101–106 | 0.33437 | 0.30461 |

Two previously nonzero queries improve under the top-100 intervention and zero decline. The three other fixed-slot arms produce equal nDCG on all 26 queries. The full-dev comparison scores only the position-ten incumbent and four challenger lists. [Preflight output and scores](../results/analysis/routed_shortlists_preflight_20261125.json)

## How much does the edit position matter?

Hold the same six challengers fixed and change which positions the oracle may edit. The starting list scores 0.38206, and ideal ordering of its same ten members scores 0.53074. All values in the table use official qrels.

| Challenger source | One change at rank ten | One change at any rank | One admission with reordering |
|---|---:|---:|---:|
| mxbai top 100 outside the starting ten | 0.39401 | 0.42378 | 0.56253 |
| mxbai ranks 101–300 | 0.38417 | 0.38915 | 0.53632 |
| E5 or BGE outside mxbai top 300 | 0.38266 | 0.38478 | 0.53314 |
| mxbai ranks 101–106 | 0.38672 | 0.39620 | 0.54442 |

Ideal selection and reordering over the starting ten and the entire union of these four shortlists scores **0.58616**. Reaching 0.60 with this starting list requires a larger or more effective shortlist. Reordering the same documents also leaves much more headroom than fixed-position edits using the same source. [Exact scope bounds](../results/analysis/routed_shortlist_edit_scope_20261125.json)

At a fixed number of extra candidates, the next audit compares a shortlist drawn entirely from the dense top 100 with two half-and-half allocations. One uses the highest mxbai ranks beyond 100 for the second half. The other interleaves three outside lists, namely RRF over mxbai ranks 101–300, unmodified mxbai ranks 101–300, and RRF over E5/BGE candidates outside mxbai top 300. Duplicate documents are skipped. In every arm the oracle can reorder the union of the starting ten and the extras.

| Extra candidates per query | All from top 100 | Half near the dense cutoff | Half from mixed outside |
|---|---:|---:|---:|
| 6 | 0.56611 | 0.55127 | 0.55336 |
| 12 | 0.59621 | 0.57975 | 0.57383 |
| 24 | 0.63242 | 0.61643 | 0.61551 |
| 48 | 0.69223 | 0.65864 | 0.66222 |
| 72 | 0.72126 | 0.71124 | 0.70403 |

At 24 extra scores, the fixed top-100 shortlist has enough label-based headroom to test a 0.60 target, with an oracle 0.01599 above the strongest measured mixed allocation at that budget. The shortlist source result holds across all five tested budgets. The next question for a same-budget scorer is how much of this headroom can be realized while recovering the first positive on queries the top 100 misses. [Per-budget results](../results/analysis/routed_shortlist_edit_scope_20261125.json)

## Relation to adaptive retrieval work

Huo et al. measure how each query's effectiveness changes with reranking depth and learn to predict depth from the initial ranking. Genc et al. compare routing to retrievers or rerankers of different costs. The corresponding RETECO comparison holds the downstream scorer fixed and measures two choices about the existing list, the source of candidate additions and which positions can change. [Huo et al., SIGIR 2026](https://doi.org/10.1145/3805712.3809953) · [Genc et al., 2026 preprint](https://arxiv.org/abs/2606.25249)

## Reproduction

```bash
python scripts/replay_accepted_membership.py --oof results/analysis/anchor_edit_scope_qwen8b_oof100_20261125.json --dev results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json --output results/analysis/accepted_membership_replay.json

python scripts/eval_routed_shortlists.py --reference results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json --routes results/analysis/adaptive_candidate_depth_dev_20261125.json --write-manifest /tmp/reteco-shortlists.json --output results/analysis/routed_shortlists_oracle_replay.json

python scripts/audit_shortlist_edit_scope.py --manifest /tmp/reteco-shortlists.json --output results/analysis/routed_shortlist_edit_scope_replay.json

python scripts/eval_routed_shortlists.py --manifest /tmp/reteco-shortlists.json --model models/Qwen3-4B-Instruct-2507 --boundary-only --checkpoint /tmp/reteco-shortlists-boundary --output results/analysis/routed_shortlists_official_dev_replay.json
```
