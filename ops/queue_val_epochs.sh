#!/usr/bin/env bash
# Rerank the 45 held-out training queries with every saved epoch of two arms,
# so the CE-only and fused validation picks can be compared.  45 queries x 100
# candidates per epoch: minutes, not hours.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/d/anaconda/python.exe
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
LOG=logs/val_epochs.log
mkdir -p logs runs
echo "[$(date +%H:%M:%S)] val epoch rerank start" | tee -a "$LOG"

for d in $DOMAINS; do
  if [ ! -s "runs/dense_mxbai_${d}_valpool.trec" ]; then
    $PY scripts/filter_run.py --run "runs/dense_mxbai_${d}_train.trec" \
      --qids results/val_qids_e2.json --domain "$d" \
      --out "runs/dense_mxbai_${d}_valpool.trec" >>"$LOG" 2>&1 || exit 1
  fi
done

run_one () {
  local tag="$1" dir="$2" mode="$3" cap="$4"
  for d in $DOMAINS; do
    local out="runs/rerank_${tag}_${d}_val.trec"
    [ -s "$out" ] && { echo "[$(date +%H:%M:%S)] skip $tag $d" >>"$LOG"; continue; }
    echo "[$(date +%H:%M:%S)] rerank $tag $d" >>"$LOG"
    $PY -u scripts/run_chunk_rerank_qvar.py --domain "$d" --split train \
      --run "runs/dense_mxbai_${d}_valpool.trec" --k 100 \
      --model "$dir" --q-mode "$mode" --q-cap-words "$cap" \
      --q-tokens 256 --win 250 --stride 192 --pool max --batch 32 \
      --out "$out" --meta "runs/rerank_${tag}_${d}_val.meta.json" >>"$LOG" 2>&1 || exit 1
  done
}

run_one lx_ep1 results/ce_ft/lx_possum/epoch1 title 256
run_one lx_ep2 results/ce_ft/lx_possum/epoch2 title 256
run_one cx_ep1 results/ce_ft/cx_qv/epoch1 full 64
run_one cx_ep2 results/ce_ft/cx_qv/epoch2 full 64
echo "[$(date +%H:%M:%S)] done" | tee -a "$LOG"
