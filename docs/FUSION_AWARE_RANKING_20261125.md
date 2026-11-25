# Train against the fused rank at ten

The earlier 4B adapter fit a standalone relevance objective on sampled lists. This experiment starts with Yuanbao-Reranker-0.6B and uses complete 30-document candidate lists. Its rank loss directly scores the equal-weight blend of the standardized reference-rank discount and standardized reranker log odds. The held-out comparison uses the same full query, two excerpts and candidate IDs across the frozen model and both trained checkpoints. [Earlier adaptation and model controls](QWEN_ADAPTATION_AND_PAIR_EVIDENCE_20261125.md)

## Training setup

Build the label-free 30-document pools from all 969 previously scored training queries using the out-of-fold reference. Exclude the 186-query evaluation set before using labels. Of the remaining queries, 611 have both a judged relevant candidate and a benchmark-zero candidate. The complete groups contain 18,330 pairs and 1,283 judged positives. Qrels assign the training targets. The 1,024-token cap reduces 16 long queries and truncates 356 document inputs. The 611 groups cover 12 domains, with the largest count in History. Group weighting reduces the influence of domains with more training queries by the square root of their frequency.

Apply rank-8 LoRA to the query projection, value projection and MLP down projection in the last eight layers. Train 589,824 parameters for two epochs at peak learning rate 0.00002. Rank the entire group by the fused score and weight pairwise loss by the absolute change in nDCG@10. A relevant document just below ten contributes when exchanged with a document above the cutoff. Add a 0.05-weighted binary loss on the unscaled yes/no log odds. The 154 updates run on four RTX 3090s in 413 seconds, with peak PyTorch allocation of 17,055 MiB. [Training metadata and checkpoint hashes](../results/analysis/yuanbao_fusion_ndcg_training_20261125.json)

| Same 30 candidates on the 186-query holdout | Frozen Yuanbao | Epoch 1 | Epoch 2 |
| --- | ---: | ---: | ---: |
| Standalone reranker | 0.29488 | 0.29231 | 0.29350 |
| Equal-weight fusion, first position open | 0.32524 | 0.32484 | 0.32491 |
| Reorder current top ten only | 0.32780 | 0.32745 | 0.32766 |

The current reference on this sample scores 0.32764. Selecting checkpoints and ranking rules on the other folds results in **0.32647**, a change of −0.00117 [−0.00327, +0.00092]. The 65 exact-reference queries score 0.32255 against 0.32633. The matched fused ranking stays near the frozen result when the fusion objective is optimized directly. This run targets the ranking objective while the small training distribution and original evidence input remain fixed. [Held-out rankings](../results/analysis/yuanbao_fusion_ndcg_holdout_20261125.json) · [Adapter logits](../results/analysis/yuanbao_fusion_ndcg_holdout_scores_20261125.tar.gz)

## Replay

Run `build_rerank_headroom_study.py --extra-per-domain 10000` and keep the rows with `split == 'train'` in a `queries` object. For preparation, use the original 186-query held-out manifest in `--holdout` and the full training candidate manifest in `--fusion-candidates`. The preparation command excludes held-out IDs and selects the 611 groups before encoding candidates.

```bash
python scripts/train_qwen_reranker_adapter.py prepare --data-root data/track1_tempo \
  --model models/Yuanbao-Reranker-0.6B --fusion-candidates "$work/fusion-train.json" \
  --holdout "$work/train.json" --max-length 1024 --output "$work/fusion-groups.json"

torchrun --standalone --nproc-per-node 4 scripts/train_qwen_reranker_adapter.py train \
  --model models/Yuanbao-Reranker-0.6B --input "$work/fusion-groups.json" \
  --objective fusion_ndcg --epochs 2 --batch-groups 1 --accumulate 2 --lr 2e-5 \
  --output "$work/fusion-checkpoints"
```

Score the original 30 candidates using `score_specialist_shortlists.py --evidence-mode dispersed --adapter` for each checkpoint. Evaluate with `eval_qwen_reranker_adapter.py` and the frozen Yuanbao scores. For exact candidate-scope checks, restrict the archived frozen 60-document score rows to the original 30 IDs.
