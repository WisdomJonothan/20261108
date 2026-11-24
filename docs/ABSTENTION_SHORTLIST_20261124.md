# Reaching new relevant documents with six comparisons

The [top-three study](HEAD_AGREEMENT_20261124.md) measured exchanges among documents already present in the list. This follow-up measures whether the same budget can expose more relevant outsiders. Throughout, the competition reference is the six-pair, one-replacement system (0.37350 official dev).

## Where shared-reference fallback reaches the remaining zeroes

Of the 130 official-dev queries with zero nDCG under the competition system, 78 have a judged relevant document in dense top 100. The previously scored 20-document anchor pool contains a positive for 29 of the 78. Only 14 of those 29 reach the fallback, because the six-pair judge accepts a replacement on the other 15. The fallback recovers seven of the fourteen. The fixed-head and confirmation variants recover the same seven. Qrels define these diagnostic groups. An accepted preference and a judged positive in the final list are separate outcomes. [Per-query routing trace](../results/analysis/zero_fallback_routing_dev_20261124.json)

## Diversifying the last three comparisons in the existing cache

Keep the first three comparisons. When all three fail the original acceptance gate, favor unseen challengers in the existing eighteen-pair cache. Fill remaining slots using the same cache order. The reference and treatment both judge six pairs in both orders and can take one swap. The gate uses model predictions only.

| Result on 969 training-fold queries | External six | Diversify after prefix abstention |
|---|---:|---:|
| Distinct challengers per query | 4.11 | 4.92 |
| Beneficial comparison for a zero-score fused query | 38 | 42 |
| Zero-score fused queries recovered | 7 | 8 |
| New zeroes | 6 | 7 |
| Domain-macro nDCG@10 | 0.28335 | 0.28351 |

The paired change is +0.00016 with within-domain 95% interval [−0.00071, +0.00110]. The gate sends 676 queries to the second stage. The last three comparisons actually change on 394 queries. [Replay results](../results/analysis/abstention_diversity_oof_20261124.json)

## Compare two candidate selectors on one shared incumbent

Keep the same first three comparisons and the same abstention rule. When routed, compare three unseen challengers from dense top 100 with the original rank-five document. The two arms use the existing best-rank union and the [held-out supervised document prior](SWAP_RISK_FOLLOWUP_20261124.md). They use the same passage selection, judge, incumbent rank, one-swap limit and budget. Prefix decisions are frozen, and both arms reuse predictions for identical pairs. Labels measure results and do not route a query.

| Candidate ceiling on 969 training-fold queries | External six | Rank union | Supervised |
|---|---:|---:|---:|
| Oracle nDCG@10 | 0.31109 | 0.31561 | 0.31743 |
| Zero-score fused queries with a beneficial comparison | 38 | 42 | 46 |

The 969-query set contains 187 fused zeroes with at least one positive in dense top 100. Prefix abstention routes 123 of the 187 to these shortlist policies. The oracle counts describe the same six available comparisons per arm and allow an ideal decision within them. The supervised oracle improves on f0 and f2 and falls slightly on f1. [Oracle results](../results/analysis/abstention_shortlist_oracle_oof_20261124.json)

### Fresh judging on f0

The 328-query f0 pilot has the largest supervised oracle improvement of the three folds. It uses 2,000 fresh prompts across both treatment arms, with cached comparisons and identical newly proposed pairs shared. New inference takes 223 seconds and peaks at 9,427 MiB of PyTorch allocation. Each individual policy uses the original twelve-prompt budget.

| Policy | nDCG@10 | Zeroes with beneficial pair → accepted → recovered | Helpful / harmful swaps |
|---|---:|---|---:|
| External six | 0.25976 | 13 → 2 → 1 | 20 / 7 |
| Rank union | 0.26011 | 18 → 6 → 4 | 22 / 11 |
| Supervised | 0.26573 | 19 → 6 → 4 | 21 / 8 |

The supervised policy gains +0.00597 over external six [−0.00186, +0.01388] and +0.00562 over rank union [−0.00152, +0.01286]. Both methods recover four zeroes, with different query IDs. One iota query accounts for +0.00505 of the domain-macro increase, since f0 contains two iota queries. The query-weighted supervised gain is +0.00058. The other two folds evaluate the identical policy. [Fresh pilot](../results/analysis/abstention_shortlist_judge_f0_20261124.json)

### Replication and pooled training-fold results

| Evaluation | External six | Rank union | Supervised | Supervised gain vs external six |
|---|---:|---:|---:|---:|
| f0, 328 queries | 0.25976 | 0.26011 | 0.26573 | +0.00597 |
| f1, 323 queries | 0.28610 | 0.28556 | 0.28655 | +0.00044 |
| f2, 318 queries | 0.30635 | 0.30596 | 0.30790 | +0.00155 |
| Combined, 969 queries | 0.28335 | 0.28313 | 0.28598 | +0.00262 |

The pooled paired interval is [−0.00135, +0.00707] against external six and [−0.00058, +0.00705] against rank union. Query-weighted gain over external six is +0.00140. Domain-resampling yields [−0.00068, +0.00673]. The policies respectively recover 7, 9 and 13 fused zeroes, and create 6, 8 and 7 new zeroes. The supervised branch has 46 zero-score queries with a beneficial available comparison, 18 with one accepted, and 13 with one finally selected. The full comparison uses 6,272 fresh prompts across both arms, about 609 seconds of judging, and 9,427 MiB of peak allocation. Each policy remains at 12 prompts per query.

Each query uses a cross-encoder whose training excludes that query. The logistic shortlist is fitted on the other folds' cached features and labels. The cross-encoders generating those training features use rotating complements that include queries from the outer evaluation fold. A fully nested estimate would regenerate training features within each outer split. Official dev uses the existing train-fitted shortlist. [Fresh replication](../results/analysis/abstention_shortlist_judge_f12_20261124.json) · [Pooled paired results](../results/analysis/abstention_shortlist_judge_oof_summary_20261124.json)

### Official-dev transfer

The 519-query dev run uses the same policy and existing training-fitted coefficients. The two treatment arms share 2,862 fresh prompts, 1.68 million input tokens and 271 seconds of judging. Each individual policy still uses twelve prompts per query.

| Policy | nDCG@10 | Fused zeroes with beneficial pair → accepted → recovered | New zeroes | Helpful / harmful swaps |
|---|---:|---|---:|---:|
| External six | 0.37350 | 23 → 12 → 9 | 1 | 33 / 10 |
| Rank union | 0.37058 | 25 → 12 → 9 | 2 | 33 / 15 |
| Supervised | 0.37154 | 30 → 14 → 12 | 4 | 36 / 15 |

Supervised changes official domain-macro nDCG by −0.00196 [−0.00907, +0.00438] versus external six and +0.00096 [−0.00205, +0.00392] versus rank union. Compared directly with external six, the supervised arm turns four competition zeroes into nonzero scores and four other queries into zeroes. Both lists leave 130 zeroes. To partition the paired nDCG difference, sum paired query deltas for each group per domain, divide by the full domain query count, and average across domains. The 130 competition zeroes contribute +0.00318. The remaining 389 queries contribute −0.00515. This identifies a concrete trade-off between recovering the first positive and preserving the positions of relevant documents already found. The project keeps the external-six competition entry. [Official-dev judgments and comparisons](../results/analysis/abstention_shortlist_judge_dev_20261124.json)

## Relation to budgeted reranking

EcoRank allocates a budget across prompting strategies and model costs. AcuRank uses estimated document ranking uncertainty to choose what to revisit near a ranking boundary. Optimal Re-Ranking Depth evaluates a signal based on an LLM's assessment of first-stage ranking quality. The experiment here measures how a preference-based gate affects access to the first judged positive within the same comparison budget. The 15 blocked queries give a concrete reason to measure relevance support directly in a follow-up. [EcoRank](https://aclanthology.org/2024.findings-acl.773/) · [AcuRank](https://arxiv.org/abs/2505.18512) · [Optimal Re-Ranking Depth](https://eprints.gla.ac.uk/383304/)

## Reproduction

Use the existing `challenger_shortlist_20261124.json` and frozen action caches. The oracle and score replays use official v1.1 qrels. Fresh scoring also reads the associated documents.

```bash
python scripts/audit_zero_fallback_routing.py --output results/analysis/zero_fallback_routing_dev_20261124.json
python scripts/replay_abstention_diversity.py --split oof --output results/analysis/abstention_diversity_oof_20261124.json
python scripts/eval_abstention_shortlist.py --folds f0 f1 f2 --output results/analysis/abstention_shortlist_oracle_oof_20261124.json
python scripts/eval_abstention_shortlist.py --folds f0 --model models/Qwen3-4B-Instruct-2507 --output results/analysis/abstention_shortlist_judge_f0_20261124.json
python scripts/eval_abstention_shortlist.py --folds f1 f2 --model models/Qwen3-4B-Instruct-2507 --output results/analysis/abstention_shortlist_judge_f12_20261124.json
python scripts/summarize_abstention_shortlist.py --inputs results/analysis/abstention_shortlist_judge_f0_20261124.json results/analysis/abstention_shortlist_judge_f12_20261124.json --output results/analysis/abstention_shortlist_judge_oof_summary_20261124.json
python scripts/eval_abstention_shortlist.py --folds dev --model models/Qwen3-4B-Instruct-2507 --output results/analysis/abstention_shortlist_judge_dev_20261124.json
```
