# Measuring evidence gained and lost in a replacement

The [fixed-budget shortlist study](ABSTENTION_SHORTLIST_20261124.md) gained +0.00318 nDCG on the competition system's zero-score dev queries and lost −0.00515 on the other queries. This study keeps its document pairs fixed and tests whether the judge can distinguish new evidence from evidence at risk of removal.

## Same-pair comparison

The control is the supervised-shortlist Direct arm. Both methods use the same Qwen3-4B-Instruct checkpoint, first three comparisons, abstention routing rule, candidate IDs, rank-five incumbent, 112-word excerpts, two document orders, and one replacement at most. The new prompt changes only the three comparisons after prefix abstention. It asks whether each passage independently provides concrete support for at least one requested part under the applicable conditions. Its four outputs are incumbent only, challenger only, both, and neither. It uses one answer token per prompt and the same count of twelve prompts per query.

The primary rule accepts a candidate when challenger-only is the maximum-probability outcome in both orders and its mean probability exceeds 0.5. It then selects the accepted candidate with highest mean probability. The conditional-preference control uses the same four outputs. It conditions on one passage qualifying and compares only the two exclusive probabilities, testing what the two extra outcomes contribute to the decision. Qrels define the retrospective outcomes. Documents without a positive judgment count as nonpositive in this analysis.

Pointwise ranking work has found useful distinctions from richer relevance labels. This comparison focuses on the joint outcomes of two documents whose replacement would change the returned list. [Beyond Yes and No](https://aclanthology.org/2024.naacl-short.31/) · [Likert or Not](https://arxiv.org/abs/2505.19334)

## Headroom from preserving known positives

The label-based veto keeps the Direct probabilities and allows only the suffix actions that would preserve a currently judged-relevant document or replace it with another judged-relevant document. The first three comparisons and unrouted queries retain the original behavior. A second oracle requires the challenger alone to be judged relevant. These are counterfactuals over existing accepted comparisons.

| Evaluation | Direct | Veto harmful swaps | Require beneficial swap | Veto gain |
|---|---:|---:|---:|---:|
| f0 | 0.26573 | 0.26737 | 0.26787 | +0.00164 |
| f1 and f2 | 0.29678 | 0.29842 | 0.29844 | +0.00164 |
| Official dev | 0.37154 | 0.37356 | 0.37356 | +0.00203 |

The dev veto keeps all 36 helpful swaps and reduces harmful swaps from 15 to 9. It prevents three of the four new zeroes. Its score is essentially even with the 0.37350 external-six competition reference. A substantive gain from the new prompt would also require accepting some useful pairs missed by Direct. [Label-based replay](../results/analysis/joint_swap_headroom_20261124.json)

## Four outcomes on the first training fold

The pilot scores 669 fixed pairs on 223 routed queries and uses 1,338 fresh prompts. Across all 328 f0 queries, the Direct control scores 0.26573. Joint support scores 0.25869, a paired change of −0.00704 [−0.01487, +0.00063]. The conditional-preference control has the same score. Direct accepts 7 of 27 challenger-only qrels pairs and 4 of 77 incumbent-only qrels pairs in the routed suffix. Joint support accepts no suffix pair; the conditional control accepts nine pairs with neither document having a positive judgment. The largest probability averaged over presentation orders is neither for 580 pairs and both for 89. All 104 gold-exclusive comparisons receive one of those two predictions. [Four-way logits and decisions](../results/analysis/joint_swap_support_f0_20261124.json)

## Isolated passage support

The next f0 control presents each passage on its own, using the same support criterion, excerpts, candidate IDs and checkpoint. It judges the shared incumbent once and each challenger once. Relative support allows any positive challenger-minus-incumbent Yes probability. Absolute support requires challenger Yes above 0.5 and incumbent Yes at or below 0.5. Both use 892 fresh prompts for the 223 routed queries, about 69 seconds of judging. The resulting policies use 10.64 prompts per query on average versus 12 for Direct.

| Policy on the same 328 f0 queries | nDCG@10 | Helpful / harmful swaps | Fused zeroes recovered / created |
|---|---:|---:|---:|
| External six | 0.25976 | 20 / 7 | 1 / 2 |
| Supervised shortlist with Direct | 0.26573 | 21 / 8 | 4 / 2 |
| Joint support | 0.25869 | 15 / 5 | 0 / 1 |
| Isolated relative support | 0.26194 | 25 / 13 | 5 / 3 |
| Isolated absolute support | 0.26136 | 19 / 7 | 3 / 1 |

Relative support accepts 15 of 27 challenger-only pairs and 13 of 77 incumbent-only pairs. Absolute support accepts 4 of 27 and 2 of 77, respectively. Isolated passages avoid the paired prompt's collapse, with a visible gain–loss trade-off. The relative and absolute policies change nDCG by −0.00379 [−0.01071, +0.00352] and −0.00437 [−0.01111, +0.00245] against Direct. These GPU comparisons use f0, including the iota query that supplied most of the shortlist's earlier f0 gain. [Isolated scores and decisions](../results/analysis/pointwise_swap_support_f0_20261124.json)

## Which annotated steps survive the replacement

For 308 f0 queries with step annotations, reconstruct the full top ten. A step is gained if a covered step first appears after the swap, and lost if its last supporting document is removed. Query step coverage is the fraction of annotated steps represented in the top ten.

| Policy | Newly covered steps | Steps lost | Domain-macro step coverage |
|---|---:|---:|---:|
| Direct | 12 | 3 | 51.97% |
| Joint support | 4 | 1 | 47.52% |
| Isolated relative support | 19 | 6 | 48.20% |
| Isolated absolute support | 10 | 2 | 48.14% |

Step counts compare with the frozen fused top ten. Isolated relative support covers more steps in raw counts, and its query-weighted coverage exceeds Direct by 0.19 percentage points. Its domain-macro coverage trails by 3.76 points, with iota alone contributing −3.85 points. This divergence follows the task's equal weighting of domains and the two iota queries in f0. [Step-level replay](../results/analysis/swap_step_coverage_f0_20261124.json)

The two non-Direct prompts end at the f0 pilot. Keeping a known-positive incumbent accounts for nearly all of the improvement available from filtering the current Direct suffix on dev. More score headroom requires finding useful comparisons that Direct declines. The reference competition configuration remains external six at 0.37350.

## Reproduction

```bash
python scripts/eval_joint_swap_support.py --model models/Qwen3-4B-Instruct-2507 --direct-report results/analysis/abstention_shortlist_judge_f0_20261124.json --output results/analysis/joint_swap_support_f0_20261124.json
python scripts/eval_pointwise_swap_support.py --model models/Qwen3-4B-Instruct-2507 --direct-report results/analysis/abstention_shortlist_judge_f0_20261124.json --output results/analysis/pointwise_swap_support_f0_20261124.json
python scripts/audit_joint_swap_headroom.py --direct-reports results/analysis/abstention_shortlist_judge_f0_20261124.json results/analysis/abstention_shortlist_judge_f12_20261124.json results/analysis/abstention_shortlist_judge_dev_20261124.json --output results/analysis/joint_swap_headroom_20261124.json
python scripts/audit_swap_step_coverage.py --reference-arm direct --reports results/analysis/joint_swap_support_f0_20261124.json results/analysis/pointwise_swap_support_f0_20261124.json --output results/analysis/swap_step_coverage_f0_20261124.json
```
