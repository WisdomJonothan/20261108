#!/usr/bin/env bash
# Dev evaluation of the epoch the deploy-config screen would have picked.
#
# The pipeline screens checkpoints on the cross-encoder's own validation nDCG
# computed at --max-len 320, while deployment re-ranks at q_tokens+win+4 = 510.
# Re-scoring the 45 held-out queries under the deployment budget prefers
# epoch 1 of lx_possum, the opposite of what the training-time screen chose.
# This script measures whether that choice matters on the official dev split:
# it re-runs ops/final_eval.sh for lx_possum/epoch1 after the GPU frees up.
#
#   bash ops/queue_lx_ep1_dev.sh          # waits for the GPU, then runs
set -u
cd /d/RETECO || exit 1
LOG=logs/lx_ep1_dev.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

PS=powershell.exe
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

gpu_free() {
  local first
  first=$($PS -NoProfile -ExecutionPolicy Bypass -File ops/gpu_busy.ps1 2>/dev/null | head -1 | cut -d' ' -f1)
  [ "$first" = "FREE" ]
}

ok=0; waited=0
while [ $ok -lt 3 ] && [ $waited -lt 240 ]; do
  if gpu_free; then ok=$((ok+1)); else ok=0; fi
  log "gpu free check: $ok/3 (waited ${waited}m)"
  [ $ok -ge 3 ] && break
  sleep 60; waited=$((waited+1))
done
if [ $ok -lt 3 ]; then log "GPU stayed busy for 4h; aborting"; exit 1; fi

log "starting final_eval lx_ep1 (lx_possum/epoch1, title, 256)"
bash ops/final_eval.sh lx_ep1 results/ce_ft/lx_possum/epoch1 title 256 1 1 >> "$LOG" 2>&1
log "done rc=$?"
