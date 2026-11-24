# Evidence in rejected replacement pairs

The [joint-support study](JOINT_SWAP_SUPPORT_20261124.md) located 27 challenger-only positive pairs among the 669 supervised-shortlist suffix comparisons in f0. Direct accepted seven. The 20 rejected pairs cover 19 distinct queries. This audit reconstructs the actual question and 112-word excerpts, reads the complete stored challenger document, and checks the official evidence-step instructions where available.

## Case review

| Finding on the 20 rejected pairs | Pairs |
|---|---:|
| The original excerpt already contains useful evidence for a requested part or sourcing step | 7 |
| The main relevant contribution is outside the excerpt | 2 |
| The excerpt contains useful evidence and omits additional concrete support | 3 |
| The full stored document provides background without clear support for the specific question | 8 |

Five rejected challengers fit in the original excerpt in full. Twelve rejected pairs have a challenger probability above 0.5 in exactly one presentation order. The classifications come from a single case-by-case review with access to qrels and model decisions. They describe evidence visibility in the stored corpus and preserve the official labels. Several incumbents also contain useful evidence even though they lack a positive qrel for this query. [Case inventory and excerpt hashes](../results/analysis/rejected_swap_evidence_inventory_f0_20261124.json) · [Case notes](../results/analysis/rejected_swap_evidence_review_f0_20261124.json)

For example, the Eastern–Western serfdom excerpt ends just as the East–West explanation begins. Its continuation gives the institutional contrast. In the UK law-firm retirement case, the original excerpt already contains concrete dates and retirement practices. The compared document also contributes relevant U.S. practices. A broader-capacity selector and a more stable comparison could help different cases. Passage aggregation and passage placement have established precedents. [PARADE](https://arxiv.org/abs/2008.09093) · [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)

## Order agreement replay

Keep the same six comparison slots, shortlist, prefix routing, and original two-order probabilities. In the routed suffix, accept when the mean challenger probability exceeds 0.5. This control removes the requirement that the challenger be the most probable choice in both orders.

| Evaluation | Original Direct | Mean-only suffix | Paired change |
|---|---:|---:|---:|
| f0, 328 queries | 0.26573 | 0.26640 | +0.00067 |
| f1 and f2, 641 queries | 0.29678 | 0.29684 | +0.00005 |
| Official dev, 519 queries | 0.37154 | 0.37237 | +0.00083 |

On dev, the mean rule admits 16 challenger-only pairs versus 9 with agreement. It also admits 14 incumbent-only pairs versus 7. Helpful and harmful final replacements change from 36/15 to 42/17; recovered and new zero counts stay at 12 and 4. The paired dev interval is [−0.00003, +0.00214]. The 0.37350 competition configuration remains the score reference. [Fixed-pair replay](../results/analysis/swap_order_gate_20261124.json)

## Longer context on the same 104 informative pairs

The diagnostic uses all 27 challenger-only and 77 incumbent-only f0 suffix pairs. Qrels define the diagnostic set. Re-run the unchanged Direct prompt, model, documents, and presentation orders with three input variants. The expanded version contains the whole stored document up to 384 words. Longer documents use a contiguous block that contains the original excerpt. Only the text extent changes. Exact duplicate inputs reuse scores.

| Variant | Accepted from 27 useful pairs | Accepted from 77 harmful pairs |
|---|---:|---:|
| Historical short window | 7 | 4 |
| Fresh short window | 6 | 4 |
| Expand only the challenger | 11 | 3 |
| Expand both documents | 10 | 3 |

The fresh short replay agrees with 103 of 104 historical decisions. Symmetric expansion admits five new useful pairs and loses one relative to its fresh short control. Four of the newly accepted pairs were historical rejections, including two with both visible and omitted evidence, one with visible evidence, and one with weak judgment-aligned support in the case review. Both pure omission cases remain below the gate even after the whole stored document is shown. This gives separate evidence of abbreviated inputs and decisions that remain difficult with longer context. The 482 scored prompts take about 63 seconds. These conditional acceptance rates describe the selected pairs. [Matched window intervention](../results/analysis/swap_context_extent_f0_20261124.json)

## Full f0 policy

Apply symmetric expansion to every one of the 669 supervised suffix pairs for 223 routed queries. Keep the 328-query evaluation set, first three comparisons, candidates, replacement slot and decision unchanged. The run uses 89 unchanged pair scores and 85 expanded diagnostic pair scores. The 495 new pairs take 990 fresh prompts, about 0.92M prompt tokens, and 166 seconds. The policy still uses 12 prompts per query, with more text in the routed suffix.

| f0 policy | nDCG@10 | Useful / harmful suffix pairs accepted | Helpful / harmful final swaps | Fused zeroes recovered |
|---|---:|---:|---:|---:|
| Short Direct | 0.26573 | 7 / 4 | 21 / 8 | 4 |
| Both sides expanded | 0.26582 | 10 / 3 | 22 / 8 | 4 |

The paired gain is +0.00009 [0.00000, +0.00028]. Thirty-five final challenger identities change, and one query changes nDCG. The prompt newly accepts four beneficial pairs on three queries. Two concern the same Grant-arrest query, which yields the only improved final replacement. The other two lose to nonpositive candidates with higher judge confidence. The longer window also loses one previously accepted helpful health pair.

For routed queries, ideal selection among the accepted pairs reaches 0.26787 with short excerpts and 0.26760 with longer excerpts. The extra positive acceptances are concentrated in history, and the lost health acceptance offsets them under equal domain weighting. The ideal score over every offered pair is 0.30479 for both. For 308 annotated f0 queries, expanded context adds one more covered step and shifts domain-macro coverage from 51.97% to 51.98%.

The full-fold result closes this context expansion at f0. The 0.37350 external-six competition configuration remains in use. The evidence points toward evaluating a stronger judge on the same rejected pairs and their controls. The candidate oracle measures the available headroom before judgment and the accepted oracle measures the headroom after it. [Full-fold results](../results/analysis/extended_swap_context_f0_20261124.json) · [Evidence-step replay](../results/analysis/extended_swap_step_coverage_f0_20261124.json)

## Reproduction

```bash
python scripts/audit_rejected_swap_evidence.py --direct-report results/analysis/abstention_shortlist_judge_f0_20261124.json --output results/analysis/rejected_swap_evidence_inventory_f0_20261124.json
python scripts/replay_swap_order_gate.py --direct-reports results/analysis/abstention_shortlist_judge_f0_20261124.json results/analysis/abstention_shortlist_judge_f12_20261124.json results/analysis/abstention_shortlist_judge_dev_20261124.json --output results/analysis/swap_order_gate_20261124.json
python scripts/eval_swap_context_extent.py --model models/Qwen3-4B-Instruct-2507 --direct-report results/analysis/abstention_shortlist_judge_f0_20261124.json --output results/analysis/swap_context_extent_f0_20261124.json
python scripts/eval_extended_swap_context.py --model models/Qwen3-4B-Instruct-2507 --direct-report results/analysis/abstention_shortlist_judge_f0_20261124.json --pair-cache results/analysis/swap_context_extent_f0_20261124.json --output results/analysis/extended_swap_context_f0_20261124.json
python scripts/audit_swap_step_coverage.py --reference-arm direct --reports results/analysis/extended_swap_context_f0_20261124.json --output results/analysis/extended_swap_step_coverage_f0_20261124.json
```
