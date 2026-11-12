#!/usr/bin/env bash
# mxbai CLS-pooling baseline (memo item 5): re-encode with --pooling cls, run dev
# retrieval, score, and aggregate, so the paper can say what pooling does.
# The cache fingerprint binds pooling, so this never reuses the mean-pool files.
set -u
cd /d/RETECO || exit 1
PY=/d/anaconda/python.exe
LOG=logs/cls_baseline.log
PREF="Represent this sentence for searching relevant passages: "
DOMS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
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

log "cls baseline queue started; waiting for cx dev queue"
for i in $(seq 1 300); do
  grep -aq "=== cx dev queue done" logs/cxdev.log 2>/dev/null && { log "cx dev queue done"; break; }
  sleep 60
done

fail=0
for d in $DOMS; do
  if [ -s "results/score_dense_mxbai_cls_${d}_dev.json" ]; then log "skip $d (already scored)"; continue; fi
  wait_gpu 240 2 || { fail=1; break; }
  log "dense cls encode+retrieve $d"
  $PY scripts/run_dense_generic.py --index-data "data/track1_tempo/$d" --split dev \
      --model models/mxbai-embed-large-v1 --tag mxbai_cls --pooling cls \
      --query-prefix "$PREF" --topk 100 --work emb \
      --out "runs/dense_mxbai_cls_${d}_dev.trec" \
      --meta "results/dense_mxbai_cls_${d}_dev.json" >> "$LOG" 2>&1 \
    || { log "FAIL dense $d"; fail=1; continue; }
  $PY scripts/score_run.py --run "runs/dense_mxbai_cls_${d}_dev.trec" \
      --data "data/track1_tempo/$d" --split dev \
      --json "results/score_dense_mxbai_cls_${d}_dev.json" >> "$LOG" 2>&1 \
    || { log "FAIL score $d"; fail=1; continue; }
  log "done $d"
done

$PY scripts/aggregate_scores.py --pattern "results/score_dense_mxbai_cls_*_dev.json" \
    --tag dense_mxbai_cls --json "results/aggregate_dense_mxbai_cls_dev.json" >> "$LOG" 2>&1 \
  || log "FAIL aggregate"

log "=== cls baseline done (fail=$fail)"
