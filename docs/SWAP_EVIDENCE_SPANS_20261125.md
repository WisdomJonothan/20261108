# Extractive evidence pointers on fixed replacement pairs

The [checkpoint study](SWAP_JUDGE_CHECKPOINT_20261124.md) found similar pair-level discrimination for the 4B and 8B Direct judges. This pilot tests the effect of letting the same Qwen3-8B checkpoint point to an original span before making the replacement decision. It uses the same 104 f0 pairs with exactly one positive qrel, including 27 challenger-only and 77 incumbent-only pairs.

## Matched test

The selector divides each original 112-word excerpt into sentence spans. It picks one existing span or abstains, using the question and applicable time, entity, and other conditions. The forward and reverse span lists are aligned back to the original offsets before averaging the probabilities over valid choices. Each pointer is an exact substring of its input. Artifacts retain its offsets and the excerpt digest.

The comparison keeps both full excerpts, the Direct prompt, two passage orders, the original acceptance rule and the Qwen3-8B checkpoint. Both pointer arms receive the same empty or nonempty indicator. The control uses the first span in each excerpt when the selector identifies an evidence span, and the treatment displays the selected span. This controls for copying a sentence into the comparison prompt. Differences in pointer length remain part of the intervention. The original Direct comparison supplies a reference with no pointer.

Existing work studies sentence evidence selection in reranking and tests whether intermediate reasoning improves relevance predictions. This experiment isolates the effect of making a source span visible in a fixed replacement comparison. [SERank](https://doi.org/10.1016/j.eswa.2026.134288) · [Rank1](https://arxiv.org/abs/2502.18418) · [Don’t “Overthink” Pointwise Reranking](https://aclanthology.org/2026.findings-emnlp.2158/)

| Pilot arm | Beneficial pairs accepted | Harmful pairs accepted | Worst-order AUC |
|---|---:|---:|---:|
| Original 4B Direct | 7/27 | 4/77 | 0.7095 |
| Original 8B Direct | 4/27 | 2/77 | 0.7114 |
| 8B with first-span pointer | 4/27 | 1/77 | 0.6349 |
| 8B with selected-span pointer | 4/27 | 1/77 | 0.6414 |

Out of 157 distinct query–document inputs, the selector abstains on 83, selects the first span on 28, and selects a later span on 46. At least one pointer differs across the controls on 62/104 comparisons, including 15/27 beneficial and 47/77 harmful pairs. The two pointer arms accept identical pairs. Relative to original 8B, they gain and lose three beneficial accepts, and gain one and lose two harmful accepts. All four accepted beneficial pairs are already accepted by the existing 4B judge. Using the difference in selector probability of a nonempty span as a score gives AUC 0.6797 on this cohort. The paired test indicates that the selector provides additional text positions, with little decision value in the duplicated-pointer format.

The selector uses 314 prompts, the first-span control 208, and the selected-span arm 124 fresh prompts after reuse of identical inputs. Judging takes about 103 seconds total across the three stages. The pointer protocol is two-stage and adds inference work per scored list. This qrels-selected pilot reports pair decisions. The full-fold and dev configurations continue with the established Direct judge. [Offsets and pair outcomes](../results/analysis/swap_evidence_spans_pilot_f0_20261125.json)

## Official-dev ceiling by editing scope

After the pilot, we reconstructed the actual 0.37350 competition result on 519 dev queries. For each depth, the pool retains that result's ten documents and fills remaining slots from dense retrieval. With binary relevance, the best fixed-position policy fills the earliest nonpositive slots with available positives. The reorder arm can also move any surviving document. The budget counts unique incoming documents, so a full reorder with zero additions keeps the exact same document membership.

| Candidate pool | Fixed, one addition | Fixed, two additions | Reorder, one addition | Unrestricted reorder |
|---|---:|---:|---:|---:|
| Protected 20 | 0.44557 | 0.45785 | 0.58072 | 0.59250 |
| Dense 100 | 0.58504 | 0.62680 | 0.69850 | 0.74462 |
| Dense 300 | 0.67603 | 0.73982 | 0.77450 | 0.85132 |
| Dense 1000 | 0.72937 | 0.82620 | 0.82043 | 0.94159 |

Reordering only the current ten gives a ceiling of 0.52238. To create credible room for the 0.60 target, the table supports testing a wider reranking stage with freedom to move existing results. The 20-document protected pool saturates at 0.59250 even under ideal judgment. On the dense top 100, preserving all other positions with one addition reaches 0.58504. Increasing candidate depth and changing positions are distinct capabilities to test in the next ranking-stage study. [Exact per-query and per-domain ceilings](../results/analysis/competition_edit_budget_20261125.json)

## Reproduction

```bash
python scripts/eval_swap_evidence_spans.py --model models/Qwen3-8B --direct-report results/analysis/abstention_shortlist_judge_f0_20261124.json --matched-8b results/analysis/swap_judge_qwen3_8b_pilot_f0_20261124.json --output results/analysis/swap_evidence_spans_pilot_f0_20261125.json
python scripts/audit_competition_edit_budget.py --output results/analysis/competition_edit_budget_20261125.json
```
