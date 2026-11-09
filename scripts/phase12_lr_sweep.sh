#!/usr/bin/env bash
# One-epoch SFT at a chosen learning rate, same schedule shape as v2m3_pool8_e1.
# Motivation: the 2-epoch run's `best` checkpoint (epoch 1, peak lr 1.5e-5) beats
# the 1-epoch 1e-5 run on every completed dev domain, so the gain looks like a
# learning-rate effect rather than an epoch effect. This isolates it.
#   LR=2e-5 TAG=v2m3_pool8_lr2e5 bash scripts/phase12_lr_sweep.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
LR=${LR:?pass LR=2e-5}
TAG=${TAG:?pass TAG=v2m3_pool8_lr2e5}
RD=results/ce_ft/$TAG
LOG=logs/train_${TAG}.log
mkdir -p logs
echo "=== 1 epoch, lr=$LR, tag=$TAG $(date +%H:%M:%S)" | tee -a "$LOG"
$PY scripts/train_ce.py --model models/bge-reranker-v2-m3 \
  --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
  --groups-prefix results/ce_train_groups --run-tag mxbai --val-frac 0.12 --val-queries 4 --val-depth 100 \
  --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
  --epochs 1 --lr "$LR" --seed 17 --tag "$TAG" --outdir results/ce_ft 2>&1 | tee -a "$LOG"
echo "TRAIN DONE $(date +%H:%M:%S)"
grep -E "val|EPOCH" "$LOG" | tail -6
