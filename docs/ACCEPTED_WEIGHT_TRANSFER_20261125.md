# Judge weight transfer on accepted lists

The previous calibration replay found an exploratory dev increase from changing the accepted-list rule. The training holdouts gave a decline for the same rule. This follow-up fixes the two weights at 30% and 60%, uses the same ten documents for each comparison, and treats each subset as a separate domain-macro calculation. [Prior calibration experiment](ANCHOR_FUSION_CALIBRATION_20261125.md)

## Matched 4B and 8B score archives

For the existing 65-query training holdout, all 35 accepted lists have full coverage in both score caches. For official dev, 80 of the 256 accepted lists have all ten documents in the archived 4B depth-20 score cache. This filter uses document IDs. It can change the mix of questions and domains represented. Require the same fixed reference document. Standardize the original frozen scores over the full 100 and each judge's scores over the ten accepted documents, then compare 30% with 60% judging weight.

| Accepted query subset | Judge | 30% judge weight | 60% judge weight | Paired difference and 95% interval |
|---|---|---:|---:|---:|
| 35 matched training queries, 12 domains | 4B | 0.28395 | 0.28109 | −0.00286 [−0.01791, +0.01171] |
| 35 matched training queries, 12 domains | 8B | 0.30292 | 0.28900 | −0.01391 [−0.03527, +0.00656] |
| 80 matched official dev queries, 13 domains | 4B | 0.34452 | 0.33023 | −0.01429 [−0.02660, −0.00239] |
| 80 matched official dev queries, 13 domains | 8B | 0.32733 | 0.34301 | +0.01568 [−0.00174, +0.03538] |

On the matched dev queries, the 8B effect exceeds the 4B effect by +0.02997 [within-domain +0.01357, +0.04714, domain bootstrap +0.00165, +0.06093]. The corresponding difference on the old training holdout is −0.01105 [−0.03253, +0.00904]. These caches support a checkpoint-by-weight comparison on the same lists. The 4B dev prompt batches come from a smaller pool, as recorded in the source runs. [Matched rankings and uncertainty](../results/analysis/accepted_weight_checkpoint_comparison_20261125.json)

## Why the normalization scope changes the effective weight

Standardizing the original system's scores over 100 documents, while standardizing the judge's scores over the accepted ten, is equivalent to giving the judge a query-specific weight with both score sources standardized over the ten. Let r be the ratio of the original system's standard deviation on the accepted ten to its standard deviation on the full pool. The equivalent judge weight for nominal weight a is a / (a + (1 − a)r). This is an instance of rank equivalence under affine score transformations. The cached 8B implementation reproduces all 256 accepted official-dev rankings exactly. [Bruch, Gai and Ingber on rank equivalence under linear score normalization](https://arxiv.org/abs/2210.11934) · [Reference-based ranking with an LLM](https://arxiv.org/abs/2506.11452)

| Standardization or equivalent 8B weight on accepted queries | Held-out full policy | Official dev full policy |
|---|---:|---:|
| Both local, fixed 30% | 0.30238 | 0.38066 |
| Both local, fixed training-median equivalent | 0.30253 | 0.38296 |
| Query-specific equivalent, or full frozen scale | 0.30821 | 0.38374 |

The training-median equivalent weight is 37.5%. Compute the fixed median for each held-out cohort from the other cohort when estimating held-out transfer. The effective weights on official dev have 10th, 50th and 90th percentiles of 25.5%, 37.9% and 52.2%. Replacing that query-specific adjustment by the fixed training median gives −0.00079 [−0.00396, +0.00238] on dev. The original 34-document abstention fallback is included in all three full-policy rows. [Computed equivalence and score sensitivity](../results/analysis/accepted_weight_checkpoint_comparison_20261125.json)

## New 4B inference on unused training queries

Draw up to 20 six-pair accepted queries per domain from the 969 out-of-fold archive, after excluding both previously used 8B cohorts. Use the seed accepted-transfer-20261125-{domain} and sorted IDs. This yields 180 queries in 12 domains. The six-pair decision and frozen score for each query come from its original fold. Use Qwen3-4B-Instruct-2507 in BF16 with its native chat template, two presentation orders, and frozen rank five as the reference. Choose the lexical 112-word passage as in the original anchor experiment. Cap words of 2,048 characters or more at 256 characters. Score 3,308 prompts in 302 seconds, with peak PyTorch allocation of 8,963 MiB. The 8B checkpoint was absent from the server's temporary storage, and download endpoints timed out in this experiment.

| New training subset, 180 accepted queries | nDCG@10 |
|---|---:|
| Six-pair starting order | 0.27778 |
| Reorder with 30% 4B weight | **0.28789** |
| Reorder with 60% 4B weight | 0.27996 |

The 60% weight changes the fresh-sample macro by −0.00793 [−0.02132, +0.00540] compared with 30%, improving 30 queries and lowering 41. The leave-one-domain-out point estimates range from −0.01175 to −0.00078. This experiment tests transfer to an alternate checkpoint on new queries. The new-query 8B replication remains pending. [New inference results and scores](../results/analysis/accepted_weight_transfer_qwen4b_20261125.json)

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_accepted_weight_transfer.py \
  --reference results/analysis/position_oracle_oof_20261124.json \
  --exclude results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --model models/Qwen3-4B-Instruct-2507 --checkpoint /tmp/reteco-accepted-weight-4b \
  --output results/analysis/accepted_weight_transfer_qwen4b_replay.json

python scripts/compare_accepted_weight_checkpoints.py --data-root data/track1_tempo \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --oof-8b results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev-8b results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-4b results/analysis/anchor_reranker_oof_sample26_20261124.json results/analysis/anchor_reranker_oof_replication39_20261124.json \
  --dev-4b results/analysis/anchor_cap20_official_dev_20261124.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --output results/analysis/accepted_weight_checkpoint_comparison_replay.json
```
