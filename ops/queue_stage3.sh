#!/usr/bin/env bash
# Ladder steps 3 and 5: (a) train with the constraint-faithful query
# representation that won the representation ablation, (b) train on the
# backfilled groups that put the previously dropped 140 queries (294 gold
# documents) back into supervision. Both arms inherit the winning loss and
# sampler, so exactly one thing changes per arm.
set -u
cd /d/RETECO || exit 1
LOG=logs/stage3.log
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

log "stage3 started; waiting for the step-supervision arms"
waited=0
while true; do
  if grep -q "=== step experiment done" logs/step_exp.log 2>/dev/null; then
    log "step experiment finished (waited ${waited}m)"; break
  fi
  sleep 60; waited=$((waited+1))
  [ $waited -ge 360 ] && { log "timeout waiting for step arms; continuing"; break; }
done

# The step arms are the paper's centrepiece, so their dev numbers are produced
# before the representation/backfill arms (same GPU budget, better order).
if [ ! -s results/STEPDEV_DONE.json ]; then
  log "step-arm dev evaluation before the cx arms"
  for t in sx_l03 sx_l10; do
    if [ -d "results/ce_ft/$t/best" ] && [ ! -s "results/aggregate_sub_${t}_dev.json" ]; then
      if wait_gpu 240 2; then
        log "final_eval $t"
        bash ops/final_eval.sh "$t" "results/ce_ft/$t/best" title 64 1 1 >> "$LOG" 2>&1 || log "FAIL final_eval $t"
      else
        log "no free GPU for $t; skipping"
      fi
    fi
  done
  echo '{"done": true}' > results/STEPDEV_DONE.json
fi

if [ ! -s results/LOSS_WINNER.json ]; then log "no loss winner; stopping"; exit 1; fi
WINNER=$($PY -c "import json;print(json.load(open('results/LOSS_WINNER.json'))['winner'] or '')")
if [ -s results/QVARIANT_WINNER.json ]; then
  QMODE=$($PY -c "import json;print(json.load(open('results/QVARIANT_WINNER.json'))['winner'] or 'title')")
  CAP=$($PY -c "import json;print(json.load(open('results/QVARIANT_WINNER.json'))['cap_words'] or 64)")
else
  QMODE=title
  CAP=64
  log "no qvariant winner; falling back to title/64"
fi
log "inheriting winner=$WINNER qmode=$QMODE cap=$CAP"

case "$WINNER" in
  lx_allpos) LOSS="--loss allpos" ;;
  lx_ndcg)   LOSS="--loss ndcg" ;;
  lx_sqrt)   LOSS="--loss possum --sampler sqrt" ;;
  *)         LOSS="--loss possum" ;;
esac
log "loss flags: $LOSS"

# ---------------------------------------------------- arm A: constraint arm
if [ -s results/ce_ft/cx_qv/train_log.jsonl ]; then
  log "cx_qv already has a log; skipping"
else
  if wait_gpu 240 2; then
    log "training cx_qv (q-mode $QMODE, cap $CAP words)"
    $PY -u scripts/train_ce.py "${BASE[@]}" $LOSS --q-mode "$QMODE" --q-cap-words "$CAP" \
      --tag cx_qv >> logs/cx_qv.log 2>&1 || log "FAIL cx_qv"
  else
    log "SKIP cx_qv (no free GPU)"
  fi
fi

# ------------------------------------------------------- arm B: backfill arm
if [ -s results/ce_ft/cx_bf/train_log.jsonl ]; then
  log "cx_bf already has a log; skipping"
else
  if wait_gpu 240 2; then
    log "training cx_bf (backfilled groups)"
    $PY -u scripts/train_ce.py "${BASE[@]}" $LOSS --groups-prefix results/ce_train_groups_bf \
      --tag cx_bf >> logs/cx_bf.log 2>&1 || log "FAIL cx_bf"
  else
    log "SKIP cx_bf (no free GPU)"
  fi
fi

# ------------------------------------------------------------ comparisons
for arm in cx_qv cx_bf; do
  if [ -s results/val_scores_${arm}_e1.json ]; then
    $PY scripts/compare_val_arms.py --a "$WINNER" --b "$arm" --epoch -1 \
      --json results/BOOT_${arm}_vs_${WINNER}.json >> "$LOG" 2>&1 || log "FAIL compare $arm"
  else
    log "no val scores for $arm; cannot compare"
  fi
done

# ------------------------------------------------------------- selection
SEL=$($PY - <<'PYEOF'
import json, os
def best(path):
    if not os.path.exists(path):
        return None
    v = None
    for ln in open(path, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("event") == "best":
            v = r.get("val_ndcg10_domain_macro")
    return v
base = json.load(open("results/LOSS_WINNER.json", encoding="utf-8"))["winner"]
cands = {"cx_qv": best("results/ce_ft/cx_qv/train_log.jsonl"),
         "cx_bf": best("results/ce_ft/cx_bf/train_log.jsonl")}
cands = {k: v for k, v in cands.items() if v is not None}
basev = best("results/ce_ft/%s/train_log.jsonl" % base)
best_tag, best_val = base, basev if basev is not None else -1.0
for k, v in cands.items():
    # val has 45 queries only (~+-0.01 noise); treat sub-band gains as ties
    if v > best_val + 0.010:
        best_tag, best_val = k, v
ranked = sorted(cands.items(), key=lambda kv: -kv[1])
runner = ranked[0][0] if ranked else base
out = {"base": base, "base_val": basev, "cands": cands,
       "selected": best_tag, "selected_val": best_val, "runner_up": runner}
json.dump(out, open("results/STAGE3_WINNER.json", "w", encoding="utf-8"), indent=1)
print("%s %s" % (best_tag, best_val))
PYEOF
)
set -- $SEL
PICK=$1
log "selected arm: $PICK (val domain-macro $2)"

# -------------------------------------------------- dev evaluation if better
if [ "$PICK" != "$WINNER" ]; then
  RUNNER=$($PY -c "import json;print(json.load(open('results/STAGE3_WINNER.json')).get('runner_up') or '')" 2>/dev/null)
  DEV_TAGS=$PICK
  if [ -n "$RUNNER" ] && [ "$RUNNER" != "$PICK" ]; then DEV_TAGS="$PICK $RUNNER"; fi
  log "dev evaluation set: $DEV_TAGS (val tie band 0.010; dev decides)"
  for t in $DEV_TAGS; do
    if [ "$t" = "cx_bf" ]; then EVM=title; else EVM=$QMODE; fi
    if wait_gpu 240 2; then
      log "running final dev evaluation for $t (q-mode $EVM, cap $CAP)"
      bash ops/final_eval.sh "$t" "results/ce_ft/$t/best" "$EVM" "$CAP" 1 1 >> "$LOG" 2>&1 || log "FAIL final_eval $t"
    else
      log "SKIP final_eval $t (no free GPU)"
    fi
  done
else
  log "no arm beat the baseline by the 0.010 val tie band; frozen system kept"
fi

log "=== stage3 done"
