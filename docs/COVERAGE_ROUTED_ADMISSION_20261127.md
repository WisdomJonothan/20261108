# Route one admission by expected coverage of the existing list

The [protected-list ceiling](FUSION_OBJECTIVE_SAME_TEN_20261127.md#headroom-of-the-fixed-list) shows that 293 of 883 training questions have no judged positive in the current ten. An ordering-only method leaves those scores at zero. Test a candidate admission criterion that estimates when the existing ten provide little evidence.

## Fixed protocol

Score the original ten plus the first 60 mxbai results with the frozen MiniLM cross-encoder and the same dispersed evidence, 192 query and 512 pair token budgets. Train a regularized relevance model on the 450-topic fit panel. Its inputs are the normalized and raw MiniLM scores, rank discounts from dense and the maintained list, and the interaction of the text score with maintained membership. Use the binary training relevance labels and weights that equalize query and domain contributions.

Nominate the best estimated outsider. Compare admitting it at rank ten when its estimated relevance exceeds the incumbent, and adding the requirement that the sum of estimated relevance across the existing ten is below one. The second quantity estimates the expected positive count by linearity. It does not assume that documents are independent. Both arms use the very same candidate and score budget. Fix all other positions. Screen both on the 242-topic panel. Select at most one with a positive mean that remains positive after omitting any domain. An added coverage condition must also beat the ungated matched policy. The 191-topic comparison uses only the chosen fixed policy against the maintained reference. Overall and domain-omission gains must remain positive to use a new official check.

The candidate scorer processes 41,881 query-document inputs. On 242 queries the maintained rank is **0.27935** and both replacement arms give **0.27881**. Unconditional replacement changes ten lists and the coverage rule changes six. Both lose the same one relevant incumbent. The 56 low-coverage estimates include 21 of the 78 reference zero-score queries. This proxy is directionally useful for diagnosis and the one-replacement ranking gives no measured gain. [Routing estimates and ranked lists](../results/analysis/coverage_routed_242_20261127.json.gz)

## Compare ordering the eligible suffix

Hold the same candidates and MiniLM logits fixed. Compare the domain-weighted logistic estimator with a shallow boosted-tree estimator on the same features. For either estimator, protect the first three maintained positions and rank the remaining seven by estimated relevance, using either only the existing membership or its union with the dense 60. For a fixed head and relevance probabilities this is the ordering that maximizes expected discounted gain in the remaining positions. Leave an entire panel out of each fit and score all 450 discovery topics. Compare against the maintained rank and require a positive effect under every domain and panel omission. Only the strongest eligible option can receive the 242 labels using a single pooled fit. The 242 have prior project use and are separate from this model fitting and selection.

| Cross-panel 450-query evaluation | Domain-macro nDCG@10 |
| --- | ---: |
| Maintained rank | 0.31347 |
| Logistic relevance, existing ten | 0.31389 |
| Logistic relevance, expanded union | 0.31498 |
| Boosted trees, existing ten | 0.31283 |
| Boosted trees, expanded union | 0.31015 |

The logistic union increases the point estimate by +0.00151 [−0.00140, +0.00459]. The smallest domain omission is +0.00058, but removing the 155-query panel gives −0.00099. All methods fail the registered stability screen. The 242 labels are outside this suffix comparison. [Cross-panel rankings](../results/analysis/ranked_candidate_suffix_450_20261127.json.gz)

## Replay

Prepare the fixed inputs with `eval_coverage_routed_admission.py prepare`, using the archived rankings, original dense training runs and official training corpus. Score the prepared inputs with `eval_fifth_ranker.py score` and the frozen MiniLM checkpoint, with batch size 32. Use the raw score cache as `--screen-scores` for `eval_coverage_routed_admission.py evaluate` or as `--scores` for `replay_ranked_candidate_suffix.py`. Both scripts require the local training judgments.
