# Changing the judge on fixed replacement pairs

The [rejected-pair audit](REJECTED_SWAP_EVIDENCE_20261124.md) found that adding context left full-fold nDCG nearly unchanged. This checkpoint comparison holds the question, document IDs, rank-five incumbent and original 112-word excerpts fixed. Its f0 pilot includes all 104 routed suffix pairs with exactly one positive qrel. Twenty-seven pairs improve nDCG if taken and seventy-seven harm it. Label selection defines this diagnostic cohort. A full-fold policy must also handle pairs with both documents on the same side of the qrels.

The strong baseline is Qwen3-4B-Instruct-2507 with the existing Direct prompt. It uses identity-aligned A/B/C scores, both presentation orders, and the original agreement rule with mean challenger probability above 0.5. The larger checkpoint uses this same protocol with its supported non-thinking chat template. A specialist control changes the scoring interface by applying the model's official pointwise yes/no protocol. [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) · [Qwen3-Reranker-4B model card](https://huggingface.co/Qwen/Qwen3-Reranker-4B)

## Specialist reranker control

Qwen3-Reranker-4B scores each of the 157 distinct query–document inputs once. It reads the same original excerpts and uses left padding and the full assistant suffix. A positive difference in Yes probabilities accepts the challenger. It accepts 12 of 27 beneficial pairs and 16 of 77 harmful pairs, versus 7 and 4 for Direct. Of these, seven beneficial and fourteen harmful accepts are new. The score-separation AUC between these two classes is 0.695 for the pointwise score gap; the Direct worst-order challenger probability gives 0.709.

The descriptive absolute-support threshold at 0.5 on each document allows six beneficial and one harmful pair. It is more conservative on this cohort. The primary pointwise gap policy ends at the pilot. The run scores 70,026 input tokens in 16.1 seconds. [Specialist decisions](../results/analysis/swap_judge_reranker4b_pilot_f0_20261124.json)

## Larger checkpoint with the same Direct protocol

Qwen3-8B scores exactly the same 104 pairs, twice per pair. It uses its non-thinking chat template and native bfloat16 weights. All five weight shards were checked against the Qwen-published SHA-256 manifest. The checkpoint index hash is `f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc`. The 208 prompts take 31.9 seconds and 16,577 MiB of peak PyTorch allocation. [Qwen-published checkpoint](https://modelscope.cn/models/Qwen/Qwen3-8B)

| Judge and decision rule | Beneficial pairs accepted | Harmful pairs accepted | AUC across the 104 exclusive pairs |
|---|---:|---:|---:|
| Qwen3-4B-Instruct-2507 Direct | 7/27 | 4/77 | 0.7095 |
| Qwen3-8B Direct | 4/27 | 2/77 | 0.7114 |
| Qwen3-Reranker-4B pointwise gap | 12/27 | 16/77 | 0.6950 |

For Direct, the AUC ranks pairs by the lower challenger probability across the two orders. The specialist uses its challenger-minus-incumbent Yes probability difference. The 8B model keeps two of the original seven beneficial accepts and admits two additional documents, both for the same Grant-arrest query. Those are the same two candidates unlocked by expanding both excerpts in the preceding study. Its two harmful accepts are newly introduced relative to the original Direct judge. Accepting a pair that passes either Direct checkpoint raises the pair count to nine beneficial and six harmful. Requiring both leaves two beneficial and zero harmful. These are pair-level overlaps on the diagnostic cohort.

The models differ in checkpoint and release recipe. The larger checkpoint trades fewer harmful accepts for fewer beneficial accepts at the fixed gate, with very similar score separation. The pair-level signal does not justify full-fold expansion. The competition reference stays at 0.37350. [Qwen3-8B votes](../results/analysis/swap_judge_qwen3_8b_pilot_f0_20261124.json)

## Reproduction

```bash
python scripts/eval_swap_judge_checkpoint.py --model models/Qwen3-8B --model-label Qwen/Qwen3-8B --disable-thinking --direct-report results/analysis/abstention_shortlist_judge_f0_20261124.json --output results/analysis/swap_judge_qwen3_8b_pilot_f0_20261124.json
python scripts/eval_swap_judge_checkpoint.py --model models/Qwen3-Reranker-4B --model-label Qwen/Qwen3-Reranker-4B --protocol reranker --direct-report results/analysis/abstention_shortlist_judge_f0_20261124.json --output results/analysis/swap_judge_reranker4b_pilot_f0_20261124.json
```
