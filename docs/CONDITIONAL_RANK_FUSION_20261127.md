# Learn which vote combinations predict relevance

The [official single-swap replay](LOCKED_ONE_SWAP_20261127.md) shows that protecting nine positions is useful only if the replacement can be selected reliably. Instead of opening a fixed slot, train a small relevance predictor using the original individual votes. Its input is the four archived ranks, the maintained output rank, counts of voters near the top, and list-overlap features. The candidate union and all four original model calls are fixed.

## Protocol fixed before scoring

Use the three archived panels with aligned individual votes and fixed references (186, 155, and 109 queries). Freeze two regularized predictors, a linear logistic model and a shallow histogram gradient boosting model. Share query-balanced and domain-balanced training weights. Treat relevance as the official binary judgment. Fit each on two panels and infer all rankings on the third without using that panel for fitting. A document absent from a source gets score zero from that source. Model correction gets one quarter of the standardized aggregate ranking score. Use the original ten-document rank discounts for the other three quarters. Separate same-ten reordering from a ranking on the union with rank one protected. Compare all four arms against the published maintained rank.

Select at most one arm by the highest cross-panel gain, requiring positive gains after dropping any panel or domain. Apply its fully trained configuration once on the separate 242-query archived population. Require a positive domain-macro gain and positive effects under every domain omission before considering another official-dev comparison. That dataset has appeared in prior studies and tests internal transfer in this experiment. No official dev labels are used to fit or select the predictor.

## Outcome

On all 450 cross-panel predictions, the maintained score is **0.31347**. The linear model scores 0.31337 inside the current ten and 0.31327 on the protected union. The tree model scores 0.31278 and 0.31310. All changes are negative. The 242 confirmation stays unopened for these models. The comparison supports testing an additional text-based signal, since this model only recombines information already represented by the four votes. [Per-query rankings and panel omissions](../results/analysis/conditional_rank_fusion_450_20261127.json.gz)
