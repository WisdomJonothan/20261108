#!/usr/bin/env bash
# Stage 4: give the step-supervision arms (sx_l03 / sx_l10) their own dev numbers.
# Rationale: the main innovation line must be reported with dev domain-macro +
# per-domain deltas + paired CI, not only with the 45-query internal split.
# Runs after the deferred loss-arm dev evaluation finishes, so it never competes
# for the single 8 GB GPU with the main chain.
set -u
cd "$(dirname "$0")/.."
LOG=logs/stage4.log
exec >>"$LOG" 2>&1
log() { echo "[$(date +%H:%M:%S)] $*"; }

WAIT_MAX=$((12 * 60))
waited=0
while [ "$waited" -lt "$WAIT_MAX" ]; do
  if grep -q "=== lossdev tail done" logs/lossdev_tail.log 2>/dev/null; then
    log "prerequisite seen: lossdev tail done"
    break
  fi
  sleep 60
  waited=$((waited + 1))
done
if [ "$waited" -ge "$WAIT_MAX" ]; then
  log "=== stage4 TIMEOUT waiting for lossdev tail done"
  exit 0
fi

log "=== stage4 start $(date +%H:%M:%S)"

for tag in sx_l03 sx_l10; do
  ck="results/ce_ft/${tag}/best"
  if [ ! -d "$ck" ]; then
    log "skip $tag: no checkpoint at $ck"
    continue
  fi
  if [ -s "results/aggregate_sub_${tag}_dev.json" ]; then
    log "skip $tag: dev aggregate already present"
    continue
  fi
  # wait for the GPU to be free (same busy check as the other queues)
  for i in $(seq 1 120); do
    out=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File ops/gpu_busy.ps1 2>/dev/null | tr -d '\r')
    case "$out" in
      FREE*) break ;;
      *) sleep 60 ;;
    esac
  done
  log "=== final_eval $tag"
  bash ops/final_eval.sh "$tag" "$ck" title 256 1 1 >> "$LOG" 2>&1 \
    || log "FAIL final_eval $tag"
done

log "=== stage4 done"
