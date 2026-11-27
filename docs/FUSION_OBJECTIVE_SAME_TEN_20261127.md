# Train the reranker for the ranking that actually gets served

The [bounded text corrections](DUAL_CROSS_ENCODER_20261127.md) compare frozen signals. This experiment fits one lightweight cross-encoder directly on the exact document set where the score will be used. The matching control learns from the same cases without the fixed rank prior in its training objective.

## Fixed design

Use the original ten, the same evidence excerpts, and the first 192 query tokens. Train on the three panels totaling 450 questions. Position one is preserved at inference and omitted from the softmax loss. Keep queries that have at least one judged positive and at least one benchmark-zero document among positions two through ten. Do not use the 242 or the 191 question judgments in the weight updates.

Start each arm from the local MiniLM checkpoint. Train for one epoch with AdamW at 2e-5, four whole-question groups per update, the same question order, and the same query weighting. Both losses use a uniform target over the judged positives below position one. The independent objective uses standardized model logits for the softmax. The fusion-aware objective adds the deployed three-times standardized rank discount before that softmax. Both arms use the identical 25% normalized correction and original head at inference.

Compare the original ranking, the frozen model, and both trained models on the 242-panel after obtaining all the scores. Advance the better trained arm only with positive improvement over the original and frozen model overall and after omitting any domain. Freeze only that arm for the 191, and measure the same comparisons there. The 242 and 191 have prior project exposure. Their labels are outside weight fitting and the 191 stay outside method selection. An official comparison requires a passing selected rule on both panels.

Both MiniLM runs use 276 informative lists and 69 optimizer updates. The maintained rank scores 0.27935, the frozen MiniLM 0.27803, the independent objective 0.27941, and the fusion objective 0.27959. The fusion objective adds +0.00156 over its frozen checkpoint and +0.00025 over the maintained ranking. That second difference reverses when leaving out a domain. The matched advantage against independence is +0.00019 [−0.00089, +0.00157]. Neither advances. [Fit records](../results/analysis/same_ten_objective_fit_20261127.json) · [Scores and ranks](../results/analysis/trained_same_ten_242_20261127.json.gz)

## Adapt the stronger cross-encoder

Use the original Mixedbread checkpoint and add rank-eight updates to the query and value projections in its last four blocks. Freeze all original parameters and initialize the added path at zero. Use the same domain-weighted group construction, one epoch and 69 optimizer steps with the fusion objective. Set the adapter learning rate to 1e-4. Merge the learned update into a local checkpoint. Compare against the exact frozen large scorer and maintained rank on the same 242 panel. Require positive effects against both under every domain omission before applying the unchanged checkpoint to the 191 confirmation. This follow-up tests adaptation of a different backbone against its own frozen state.
