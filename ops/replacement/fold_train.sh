#!/bin/bash
# usage: fold_train.sh <fold: f0|f1|f2|fall>
set -e
cd ~/RETECO
F=$1
mkdir -p logs /tmp/reteco/ce_ft
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u scripts/train_ce.py \
  --model models/bge-reranker-v2-m3 \
  --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
  --groups-prefix results/ce_folds/$F/g_$F \
  --run-tag mxbai --val-frac 0.12 --val-queries 4 --val-depth 100 \
  --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
  --epochs 2 --lr 1.5e-5 --seed 17 --tag ${F}_ce --outdir /tmp/reteco/ce_ft/$F \
  > logs/${F}_ce.log 2>&1 &
echo "launched $F pid $!"
