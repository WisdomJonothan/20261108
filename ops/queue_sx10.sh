#!/usr/bin/env bash
# Second step-supervision arm (lambda = 1.0) plus its dev evaluation, then the
# canonical "step experiment done" marker so that queue_stage3.sh can proceed.
set -u
cd /d/RETECO || exit 1
PY=/d/anaconda/python.exe
LOG=logs/sx_l10_run.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

DOMS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"

wait_gpu() {  # max_minutes, consecutive_checks
  local maxm=$1 need=$2 ok=0 waited=0
  while true; do
    if powershell.exe -NoProfile -Command "& 'D:\RETECO\ops\gpu_busy.ps1'" 2>/dev/null | grep -q '^FREE'; then
      ok=$((ok+1)); log "gpu free check: $ok/$need (waited ${waited}m)"
      [ $ok -ge $need ] && return 0
    else
      ok=0
    fi
    sleep 60; waited=$((waited+1))
    if [ $waited -ge $maxm ]; then log "gpu wait timeout"; return 1; fi
  done
}

log "sx_l10 (lambda=1.0) run started"
if [ -d results/ce_ft/sx_l10/best ]; then
  log "sx_l10 checkpoint already exists; skipping training"
else
  wait_gpu 240 2 || exit 1
  log "START training sx_l10"
  $PY scripts/train_ce.py --domains $DOMS --groups-prefix results/ce_train_groups \
      --model models/bge-reranker-v2-m3 --run-tag mxbai \
      --epochs 2 --lr 1.5e-5 --warmup-frac 0.08 --freeze-layers 8 \
      --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
      --val-frac 0.12 --val-queries 4 --val-depth 100 --seed 17 \
      --val-metric domain_macro --report-steps --dump-val-scores \
      --q-mode title --q-cap-words 64 \
      --val-qids results/val_qids_e2.json \
      --step-targets results/ce_step_targets --step-lambda 1.0 \
      --tag sx_l10 >> logs/sx_l10.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && { log "FAIL training rc=$rc"; exit 1; }
  log "DONE training sx_l10"
fi

if [ -s results/aggregate_sub_sx_l10_dev.json ]; then
  log "dev aggregate already exists; skipping final_eval"
else
  wait_gpu 240 2 || exit 1
  log "final_eval sx_l10"
  bash ops/final_eval.sh sx_l10 results/ce_ft/sx_l10/best title 64 1 1 >> "$LOG" 2>&1 \
    || { log "FAIL final_eval sx_l10"; exit 1; }
  log "DONE final_eval sx_l10"
fi

echo "=== step experiment done" >> logs/step_exp.log
log "=== sx_l10 run done (marker written for stage3)"
