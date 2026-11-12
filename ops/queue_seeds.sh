#!/usr/bin/env bash
# Seed replication for the query-representation arm (cx_qv: full query, 64 words).
# Usage:  bash ops/queue_seeds.sh 23 29
# Each seed: train -> dev final_eval with the same q-mode/cap/fusion as the arm.
set -u
cd /d/RETECO || exit 1
PY=/d/anaconda/python.exe
LOG=logs/seeds.log
DOMS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
QMODE=full; CAP=64
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_gpu() {
  local maxm=$1 need=$2 ok=0 waited=0
  while true; do
    if powershell.exe -NoProfile -Command "& 'D:\RETECO\ops\gpu_busy.ps1'" 2>/dev/null | grep -q '^FREE'; then
      ok=$((ok+1)); log "gpu free check: $ok/$need (waited ${waited}m)"
      [ $ok -ge $need ] && return 0
    else ok=0; fi
    sleep 60; waited=$((waited+1))
    [ $waited -ge $maxm ] && { log "gpu wait timeout"; return 1; }
  done
}

for s in "$@"; do
  tag="cx_qv_s$s"
  if [ ! -d "results/ce_ft/$tag/best" ]; then
    wait_gpu 240 2 || exit 1
    log "START $tag (seed $s, q-mode $QMODE cap $CAP)"
    $PY scripts/train_ce.py --domains $DOMS --groups-prefix results/ce_train_groups \
        --model models/bge-reranker-v2-m3 --run-tag mxbai \
        --epochs 2 --lr 1.5e-5 --warmup-frac 0.08 --freeze-layers 8 \
        --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
        --val-frac 0.12 --val-queries 4 --val-depth 100 --seed "$s" \
        --val-metric domain_macro --dump-val-scores \
        --q-mode "$QMODE" --q-cap-words "$CAP" \
        --val-qids results/val_qids_e2.json \
        --tag "$tag" >> "logs/$tag.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && { log "FAIL train $tag rc=$rc"; continue; }
    log "DONE train $tag"
  else
    log "$tag checkpoint already exists"
  fi

  if [ -s "results/aggregate_sub_${tag}_dev.json" ]; then
    log "$tag dev aggregate exists; skipping final_eval"
  else
    wait_gpu 240 2 || exit 1
    log "final_eval $tag"
    bash ops/final_eval.sh "$tag" "results/ce_ft/$tag/best" "$QMODE" "$CAP" 1 1 >> "$LOG" 2>&1 \
      || { log "FAIL final_eval $tag"; continue; }
    log "DONE final_eval $tag"
  fi
done
log "=== seeds done"
