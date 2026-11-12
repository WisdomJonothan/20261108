#!/usr/bin/env bash
# Dev-evaluate the two stage-3 arms whatever the 45-query val tie band says.
# The step-arm lesson (18:05) was that this split can rank arms backwards, so
# "val says it is a tie" is not a reason to skip the dev measurement.
set -u
cd /d/RETECO || exit 1
LOG=logs/cxdev.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_gpu() {
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

log "cx dev queue started; waiting for === stage3 done"
for i in $(seq 1 240); do
  if grep -aq "=== stage3 done" logs/stage3.log 2>/dev/null; then log "stage3 done detected"; break; fi
  sleep 60
done

run_one() {  # tag qmode
  local t=$1 m=$2
  if [ -s "results/aggregate_sub_${t}_dev.json" ]; then
    log "$t already has a dev aggregate; skipping"
    return 0
  fi
  wait_gpu 240 2 || return 1
  log "final_eval $t (q-mode $m, cap 64 words)"
  bash ops/final_eval.sh "$t" "results/ce_ft/$t/best" "$m" 64 1 1 >> "$LOG" 2>&1 \
    || { log "FAIL final_eval $t"; return 1; }
  log "DONE final_eval $t"
}

run_one cx_qv full
run_one cx_bf title

log "=== cx dev queue done"
