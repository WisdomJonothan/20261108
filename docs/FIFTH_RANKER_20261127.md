# Test a fifth vote from a separate cross-encoder

The [conditional fusion check](CONDITIONAL_RANK_FUSION_20261127.md) reuses all four original rankings and its models show no mean gain. This experiment introduces a separately trained DeBERTa cross-encoder, [mxbai-rerank-large-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1/tree/98f655841d5caf0b16eaff79c2b4ca109d920d17), at revision `98f655841d5caf0b16eaff79c2b4ca109d920d17`.

## Fixed protocol

Score the union of the four existing lists and the maintained ten. Use the original question and query-matched dispersed evidence, with a 512-token pair budget and up to 192 query tokens before joint truncation. Keep the checkpoint frozen. Compare a new fifth reciprocal-rank vote at offset 60, a quarter-weight score for reordering the same ten, and two bounded uses. The head exchange requires the replacement to be in the new model's top three and the original 8B's top five. The tail exchange requires the replacement to beat the rank-ten incumbent under both scorers. The two exchanges never alter other ranks. A duplicated pointwise vote is the matched five-vote control. Both vote arms preserve the maintained first document.

The discovery population is the three archived panels totaling 450 questions. Open only the one fixed candidate with the largest positive gain that survives every panel and domain omission on a separate 242-query internal transfer. A fifth-vote policy must also exceed the duplicate-vote control at discovery. Evaluate only the locked candidate and the reference on the transfer labels. Use fixed query IDs from the existing archives, all 13 available domains, and paired uncertainty. No further use of official dev is authorized by a discovery-only gain.

## Discovery on 450 training questions

| Use of the additional score | Domain-macro nDCG@10 |
| --- | ---: |
| Maintained rank | 0.31347 |
| Duplicate the 8B pointwise vote | 0.30945 |
| Fifth vote from the new cross-encoder | 0.31421 |
| Reorder the same ten with a quarter-weight score | **0.31614** |
| One head exchange | 0.30746 |
| One tail exchange | 0.31211 |

The selected same-ten rule gains **+0.00267** [ +0.00064, +0.00473 ] with a paired query bootstrap. It improves 98 and worsens 72 questions. The minimum gains after omitting a domain or a panel are +0.00204 and +0.00254. The separate fifth vote beats duplication by +0.00476. Its effect versus the maintained rank reverses under some domain omissions. The same-ten rule alone proceeds to internal transfer at its fixed quarter weight and original input truncation.

For each existing top-ten position, use the rank discount `1/log2(1+rank)` as the prior. Add the new model's standardized score to three times the standardized prior, with both normalizations on those same ten documents. The blend keeps membership and guarantees retention of rank one. With population z-normalization over ten documents, the difference between any two normalized model scores has magnitude at most sqrt(20), or 4.47214. The minimum prior gap times three between first place and any other place is 5.33630. All 450 output lists preserve this first position, and 448 change order below it. The positive effect is therefore attributable to ordering existing results below rank one.

## Locked 242-query transfer

The same-ten rule scores **0.27929** versus **0.27935** for the maintained rank. The paired change is −0.00005 [−0.00221, +0.00204], with 41 gains and 44 declines. The minimum domain-omission change is −0.00071. The official dev labels stay outside this comparison.

## Keep the end of long questions

The fixed 192-token prefix clips 168 of the 450 discovery questions. The removed text includes an additional question mark on 111 and a four-digit year on 48. Before scoring the alternative, freeze a 64-token prefix plus the final 128 tokens, with the same model, evidence, candidate IDs, and 25% correction. Reuse exactly the same scores for shorter queries. Compare only the maintained rank and the two input policies on the fixed ten. Require the new input to beat both controls across the 450 and every domain omission before applying the same rule on the 191-query panel. Those 191 questions have prior project exposure and were held outside the current fifth-ranker and rank-only screens. They need only their existing ten documents scored because the locked correction uses no other candidate score.

The same-ten prefix-plus-ending version reaches **0.31643**, a change of +0.00296 [ +0.00105, +0.00496 ] from the maintained rank. Its matched difference against prefix-only is +0.00029 [−0.00064, +0.00122]. Omitting genealogy reverses the latter contrast to −0.00034. The input change fails the matched transfer screen. [Cached scores and matched rankings](../results/analysis/head_tail_ranking_450_20261127.json.gz)

## Replay

```text
python scripts/eval_fifth_ranker.py prepare --archive-dir results/analysis --data-root data/track1_tempo --output discovery-inputs.json.gz
python scripts/eval_fifth_ranker.py score --inputs discovery-inputs.json.gz --model models/mxbai-rerank-large-v1-98f6558 --output discovery-scores.json.gz
python scripts/eval_fifth_ranker.py evaluate --scores discovery-scores.json.gz --data-root data/track1_tempo --output discovery-result.json.gz
```

For internal transfer, add `--confirmation` when preparing inputs and pass `--selection discovery-result.json.gz` when evaluating. Excerpt text stays in the local input cache. The exported results include the candidate identities, cached logits and evaluated rankings.
