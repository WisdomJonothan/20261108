#!/usr/bin/env bash
# Ladder step 4 (user memo m00374, innovation line A): multi-evidence ranking
# supervision on top of the winning ordinary-ranking baseline.
#
# Reads results/LOSS_WINNER.json (written by stage 2), so the step-supervision
# arms inherit whatever loss / sampler won the four-arm comparison, then adds
#     L_step = -(1/m) sum_j log sum_{d in P_j} p(d)
# over the official step->document annotations. All arms keep the same pinned
# train-internal validation split, the same preprocessing and seed 17; dev is
# still untouched (dev re-ranking happens later, once, for the final system).
set -u
cd /d/RETECO || exit 1
LOG=logs/step_exp.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

PY=/d/anaconda/python.exe
PS=powershell.exe
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DOMS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
BASE=(--domains $DOMS --groups-prefix results/ce_train_groups
      --model models/bge-reranker-v2-m3 --run-tag mxbai
      --epochs 2 --lr 1.5e-5 --warmup-frac 0.08 --freeze-layers 8
      --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32
      --val-frac 0.12 --val-queries 4 --val-depth 100 --seed 17
      --val-metric domain_macro --report-steps --dump-val-scores)

gpu_free() {
  local first
  first=$($PS -NoProfile -ExecutionPolicy Bypass -File ops/gpu_busy.ps1 2>/dev/null | head -1 | cut -d' ' -f1)
  [ "$first" = "FREE" ]
}

wait_gpu() {
  local cap=${1:-240} need=${2:-2} ok=0 waited=0
  while [ $ok -lt $need ] && [ $waited -lt $cap ]; do
    if gpu_free; then ok=$((ok+1)); else ok=0; fi
    log "gpu free check: $ok/$need (waited ${waited}m)"
    [ $ok -ge $need ] && break
    sleep 60; waited=$((waited+1))
  done
  [ $ok -ge $need ]
}

# ------------------------------------------------ 1. wait for the winner
# stage 2 writes LOSS_WINNER.json and then immediately evaluates the query
# representations on that same GPU, so wait for its *last* artifact to avoid
# running two model-loaded jobs at once. Fall back after 90 minutes in case
# that evaluation failed.
waited=0
while true; do
  if [ -s results/QVARIANT_WINNER.json ]; then log "stage2 finished (waited ${waited}m)"; break; fi
  if [ -s results/LOSS_WINNER.json ] && [ -z "$(find results/LOSS_WINNER.json -newermt '-90 minutes' 2>/dev/null)" ]; then
    log "no qvariants file 90m after the winner; proceeding anyway"; break
  fi
  sleep 60; waited=$((waited+1))
  [ $waited -ge 600 ] && { log "timeout waiting for stage2"; exit 1; }
done


WINNER=$($PY -c "import json;print(json.load(open('results/LOSS_WINNER.json'))['winner'] or '')")
if [ -z "$WINNER" ]; then log "no winner recorded; stopping"; exit 1; fi
log "inheriting configuration from $WINNER"

case "$WINNER" in
  lx_allpos) LOSS=(--loss allpos) ;;
  lx_ndcg)   LOSS=(--loss ndcg) ;;
  lx_sqrt)   LOSS=(--loss possum --sampler sqrt) ;;
  *)         LOSS=(--loss possum) ;;
esac
log "loss flags: ${LOSS[*]}"

run_arm() {  # tag, lambda
  local tag=$1 lam=$2
  if [ -d "results/ce_ft/$tag/best" ]; then log "skip $tag (exists)"; return; fi
  if ! wait_gpu 240 2; then log "GPU busy before $tag; stopping"; exit 1; fi
  log "START $tag (lambda=$lam)"
  $PY scripts/train_ce.py "${BASE[@]}" "${LOSS[@]}" \
      --val-qids results/val_qids_e2.json \
      --step-targets results/ce_step_targets --step-lambda "$lam" \
      --tag "$tag" >> "logs/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "FAIL $tag rc=$rc"; return; fi
  $PY - "$tag" <<'PY' 2>>"$LOG"
import json, sys
tag = sys.argv[1]
rows = [json.loads(l) for l in open("results/ce_ft/%s/train_log.jsonl" % tag, encoding="utf-8")]
ep = [r for r in rows if r.get("event") == "epoch"]
best = [r for r in rows if r.get("event") == "best"][-1]
print("[summary] %s: " % tag +
      " ".join("e%d dm=%.4f mean=%.4f step_cov=%.3f multi10=%.3f" %
               (r["epoch"], r["val_ndcg10_domain_macro"], r["val_ndcg10"],
                r.get("step_hit_rate", -1), r.get("q_with_2pos_top10", -1)) for r in ep) +
      " | best epoch=%s score=%.4f" % (best.get("epoch"), best.get("score", 0)))
PY
  log "DONE $tag"
}

run_arm sx_l03 0.3
run_arm sx_l10 1.0

log "=== step experiment done"
