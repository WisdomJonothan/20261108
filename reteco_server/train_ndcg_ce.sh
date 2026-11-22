#!/bin/bash
# Train a cross-encoder on the full 969-query train split with the listwise
# lambda-rank nDCG@10 loss (--loss ndcg), i.e. the metric-aligned objective,
# then dump the checkpoint for dev reranking.
set -e
cd /homeC/wujiaheng/RETECO
export CUDA_VISIBLE_DEVICES=${GPU:-0}
.venv/bin/python -u scripts/train_ce.py \
  --model models/bge-reranker-v2-m3 \
  --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
  --groups-prefix results/ce_folds/fall/g_all \
  --run-tag mxbai \
  --val-frac 0.12 --val-queries 4 --val-depth 100 \
  --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 \
  --pool-chunk 32 --epochs 2 --lr 1.5e-5 --seed 17 \
  --loss ndcg \
  --tag ndcg_ce --outdir /tmp/reteco/ce_ft/ndcg
echo NDcg_CE_TRAIN_DONE
