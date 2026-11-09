#!/usr/bin/env bash
# P1 #2: seed replication of the SFT gain.
# Two more single-epoch SFT runs (seeds 23, 29), same config as v2m3_pool8_e1,
# then dev rerank + the frozen z 1:1 arm (freeze_sft_arm.sh).
#
#   bash scripts/phase11_seed_repl.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DOMS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
LOG=logs/phase11_seed_repl.log
exec >>"$LOG" 2>&1
echo "=== seed repl start $(date +%H:%M:%S)"

for s in 23 29; do
  tag=v2m3_s${s}
  if [ ! -s "results/ce_ft/$tag/best/model.safetensors" ]; then
    $PY scripts/train_ce.py --domains $DOMS --groups-prefix results/ce_train_groups \
      --model models/bge-reranker-v2-m3 --run-tag mxbai --tag "$tag" \
      --epochs 1 --lr 1e-5 --warmup-frac 0.08 --freeze-layers 8 \
      --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
      --val-frac 0.12 --val-queries 4 --val-depth 100 --seed "$s" \
      || { echo "FAIL train $s"; continue; }
  fi
  $PY scripts/check_ce_model.py --tag "$tag" --expect v2-m3 --params 567.8 \
    || { echo "FAIL model check $s"; continue; }
  CKPT="results/ce_ft/$tag/best" TAG=sft_s${s} bash scripts/phase9_dev_rerank.sh
  TAG=sft_s${s} bash scripts/freeze_sft_arm.sh || { echo "FAIL freeze $s"; continue; }
  echo "--- seed $s done $(date +%H:%M:%S)"
done
echo "=== seed repl done $(date +%H:%M:%S)"
