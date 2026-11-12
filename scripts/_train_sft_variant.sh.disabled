#!/usr/bin/env bash
# Generic SFT runner: same data/preprocessing as v2m3_pool8_e1, chosen epochs/lr/seed.
#   EPOCHS=2 LR=1.5e-5 SEED=23 TAG=v2m3_pool8_e2s23 bash scripts/train_sft_variant.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
EPOCHS=${EPOCHS:?pass EPOCHS=1|2|3}
LR=${LR:?pass LR=1.5e-5}
SEED=${SEED:?pass SEED=17|23|29}
TAG=${TAG:?pass TAG=v2m3_pool8_<variant>}
RD=results/ce_ft/$TAG
[ -e "$RD" ] && { echo "refusing: $RD exists"; exit 1; }
LOG=logs/train_${TAG}.log
mkdir -p logs
{
  echo "=== epochs=$EPOCHS lr=$LR seed=$SEED tag=$TAG $(date +%H:%M:%S)"
  $PY scripts/train_ce.py --model models/bge-reranker-v2-m3 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
    --groups-prefix results/ce_train_groups --run-tag mxbai --val-frac 0.12 --val-queries 4 --val-depth 100 \
    --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
    --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" --tag "$TAG" --outdir results/ce_ft
  echo "TRAIN DONE $(date +%H:%M:%S)"
} 2>&1 | tee -a "$LOG"
grep -E "\"epoch\"" "$LOG" | tail -8
