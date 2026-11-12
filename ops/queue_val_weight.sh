#!/usr/bin/env bash
# Choose the CE fusion weight on the 45 held-out training queries (val), then
# read the dev result of that weight from the probe.  Runs on CPU-free GPU time:
# 45 queries x 100 candidates per model.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/d/anaconda/python.exe
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
LOG=logs/val_weight.log
mkdir -p logs runs
echo "[$(date +%H:%M:%S)] val weight sweep start" | tee -a "$LOG"

for d in $DOMAINS; do
  if [ ! -s "runs/dense_mxbai_${d}_valpool.trec" ]; then
    $PY scripts/filter_run.py --run "runs/dense_mxbai_${d}_train.trec" \
      --qids results/val_qids_e2.json --domain "$d" \
      --out "runs/dense_mxbai_${d}_valpool.trec" >>"$LOG" 2>&1 || exit 1
  fi
done

run_model () {
  local tag="$1" mode="$2" cap="$3"
  for d in $DOMAINS; do
    local out="runs/rerank_${tag}_${d}_val.trec"
    [ -s "$out" ] && { echo "[$(date +%H:%M:%S)] skip $tag $d" >>"$LOG"; continue; }
    echo "[$(date +%H:%M:%S)] rerank $tag $d" >>"$LOG"
    $PY -u scripts/run_chunk_rerank_qvar.py --domain "$d" --split train \
      --run "runs/dense_mxbai_${d}_valpool.trec" --k 100 \
      --model "results/ce_ft/${tag}/best" --q-mode "$mode" --q-cap-words "$cap" \
      --q-tokens 256 --win 250 --stride 192 --pool max --batch 32 \
      --out "$out" --meta "runs/rerank_${tag}_${d}_val.meta.json" >>"$LOG" 2>&1 || exit 1
  done
}

run_model lx_possum title 256
run_model cx_qv full 64
$PY scripts/val_weight_sweep.py >>"$LOG" 2>&1
echo "[$(date +%H:%M:%S)] done" | tee -a "$LOG"
