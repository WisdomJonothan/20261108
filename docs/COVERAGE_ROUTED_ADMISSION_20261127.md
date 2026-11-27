# Route one admission by expected coverage of the existing list

The [protected-list ceiling](FUSION_OBJECTIVE_SAME_TEN_20261127.md#headroom-of-the-fixed-list) shows that 293 of 883 training questions have no judged positive in the current ten. An ordering-only method leaves those scores at zero. Test a candidate admission criterion that estimates when the existing ten provide little evidence.

## Fixed protocol

Score the original ten plus the first 60 mxbai results with the frozen MiniLM cross-encoder and the same dispersed evidence, 192 query and 512 pair token budgets. Train a regularized relevance model on the 450-topic fit panel. Its inputs are the normalized and raw MiniLM scores, rank discounts from dense and the maintained list, and the interaction of the text score with maintained membership. Use the binary training relevance labels and weights that equalize query and domain contributions.

Nominate the best estimated outsider. Compare admitting it at rank ten when its estimated relevance exceeds the incumbent, and adding the requirement that the sum of estimated relevance across the existing ten is below one. The second quantity estimates the expected positive count by linearity. It does not assume that documents are independent. Both arms use the very same candidate and score budget. Fix all other positions. Screen both on the 242-topic panel. Select at most one with a positive mean that remains positive after omitting any domain. An added coverage condition must also beat the ungated matched policy. The 191-topic comparison uses only the chosen fixed policy against the maintained reference. Overall and domain-omission gains must remain positive to use a new official check.
