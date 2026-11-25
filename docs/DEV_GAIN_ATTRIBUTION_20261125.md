# Where the RETECO dev gains come from

The official dev scores are domain-macro nDCG@10 over 519 queries in 13 equally weighted domains. The original system scored 0.1147 with the official BM25 baseline, 0.2943 with mxbai dense retrieval, 0.3249 with dense and the zero-shot BGE cross-encoder, and 0.35810 after task fine-tuning. The matched zero-shot comparison estimates a +0.0332 gain from task fine-tuning. [Baseline and fine-tuning controls](PAPER_4PAGE.md)

## Attribution across the later systems

For each query, subtract the relevance contribution of each document at its old rank and add the contribution at its new rank. Keep contributions from admitted, removed, and retained documents. Sum within each query and average across domains. The three columns sum to the paired change in score. The rank 1–3 column is an overlapping view of the same change.

| Transition | Scores before and after | Paired change [95% interval] | Admitted | Removed | Retained, rank changes | Contribution at ranks 1–3 |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned fusion to six-pair | 0.35810 → 0.37350 | +0.01540 [+0.00635, +0.02570] | +0.01874 | −0.00334 | 0 | +0.01322 |
| Six-pair to depth-100 8B on abstentions | 0.37350 → 0.38206 | +0.00855 [+0.00046, +0.01733] | +0.00797 | −0.00434 | +0.00492 | +0.00300 |
| Reorder the accepted top tens with 8B | 0.38206 → 0.38314 | +0.00109 [−0.00678, +0.00842] | 0 | 0 | +0.00109 | +0.00003 |

Six-pair contributes about 62% of the measured increase from 0.35810 to 0.38314. The rank 1–3 subtotal accounts for approximately 86% of the six-pair gain. This is consistent with the action policy retaining an incumbent unless a challenger passes both display orders. In the abstention fallback, better placement of retained documents contributes +0.00492, and changes in document membership contribute +0.00363 net. After the last reorder, the rank 4–6 subtotal is +0.00352 and the rank 7–10 subtotal is −0.00246.

Mean annotated step coverage rises from 59.26% to 61.39% to 62.62% across the first three systems. It remains 62.62% under the final order change. Step coverage counts each step once whenever any document in the top ten supports it. [Six-pair setup](POSITION_REPAIR_20261124.md) · [Full 8B comparison](ANCHOR_OFFICIAL_DEV_20261125.md)

## Reordering a prefix of an accepted list

Use the cached 8B order but permute only the documents in the first k positions of an accepted six-pair list. The original lower-ranked positions retain their documents and order in this control. Apply the same depth-100 fallback to queries on which six-pair abstains. Pick k by nDCG on the 65 held-out training queries and evaluate the selected value on official dev from the archives.

| Positions eligible for reordering | Held-out training 65 | Official dev 519 |
|---|---:|---:|
| First 1, equivalent to the fallback | 0.29960 | 0.38206 |
| First 3 | 0.30060 | 0.37953 |
| First 5 | 0.29722 | 0.37995 |
| First 8 | 0.29834 | 0.38332 |
| First 10, selected on held-out training | **0.30441** | **0.38314** |

The eight-position official-dev score is an exploratory comparison. Its difference from the fallback is +0.00127 [−0.00638, +0.00848]. Restricting reordering to three or five positions lowers the official-dev point estimate. The pattern gives a controlled distinction between admitting relevant documents near the top and reordering documents that have already been accepted. [Prefix estimates and exact rank attribution](../results/analysis/accepted_prefix_reorder_20261125.json)

```bash
python scripts/audit_accepted_prefix_reorder.py --data-root data/track1_tempo --oof results/analysis/anchor_edit_scope_qwen8b_oof100_20261125.json --dev results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json --output results/analysis/accepted_prefix_reorder_replay.json
```
