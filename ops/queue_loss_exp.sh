#!/usr/bin/env bash
# Experiment ladder steps 1-3 (user memo m00374): objective alignment.
#
#   1. wait until no other process is using the GPU for a model-loaded job
#      (ops/gpu_busy.ps1; memory alone is useless under WDDM)
#   2. pin ONE train-internal validation split so every arm sees the same
#      held-out train queries (results/val_qids_e2.json)
#   3. run four arms, all 2 epochs / seed 17 / identical preprocessing, so the
#      only difference is the training objective or the sampling policy:
#        lx_possum  current loss, -log sum(P)          (control)
#        lx_allpos  mean over positives of -log p(d)
#        lx_ndcg    LambdaRank with nDCG@10 swap weights
#        lx_sqrt    current loss + sqrt-balanced domain sampling
#      every arm selects `best` by domain-macro nDCG@10 on the pinned split.
#
# No dev qrels are read here; dev re-ranking happens later, only for the winner.
set -u
cd /d/RETECO || exit 1
LOG=logs/loss_exp.log
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
      --val-metric domain_macro --dump-val-scores)

# ------------------------------------------------------------ helpers
gpu_free() {
  local first
  first=$($PS -NoProfile -ExecutionPolicy Bypass -File ops/gpu_busy.ps1 2>/dev/null | head -1 | cut -d' ' -f1)
  [ "$first" = "FREE" ]
}

wait_gpu() {  # $1 = max minutes, $2 = consecutive free checks needed
  local cap=${1:-240} need=${2:-3} ok=0 waited=0
  while [ $ok -lt $need ] && [ $waited -lt $cap ]; do
    if gpu_free; then ok=$((ok+1)); else ok=0; fi
    log "gpu free check: $ok/$need (waited ${waited}m)"
    [ $ok -ge $need ] && break
    sleep 60; waited=$((waited+1))
  done
  [ $ok -ge $need ]
}

# ---------------------------------------------------------------- 1. wait
if ! wait_gpu 240 3; then
  log "GPU stayed busy for 4h; aborting without starting (no OOM race)"
  exit 1
fi

# ------------------------------------------------------- 2. pin val split
if [ ! -s results/val_qids_e2.json ]; then
  log "dumping pinned validation split"
  $PY scripts/train_ce.py "${BASE[@]}" --tag valpin --dump-val-qids results/val_qids_e2.json \
    >> logs/valpin.log 2>&1 || { log "FAIL valpin"; exit 1; }
  log "pin done: $(wc -c < results/val_qids_e2.json) bytes"
fi

# ------------------------------------------------------------- 3. arms
run_arm() {  # tag, extra args...
  local tag=$1; shift
  if [ -d "results/ce_ft/$tag/best" ]; then log "skip $tag (exists)"; return; fi
  if ! wait_gpu 180 2; then log "GPU busy before $tag; stopping"; exit 1; fi
  log "START $tag"
  $PY scripts/train_ce.py "${BASE[@]}" --val-qids results/val_qids_e2.json \
      --tag "$tag" "$@" >> "logs/$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then log "FAIL $tag rc=$rc"; return; fi
  $PY - "$tag" <<'PY' 2>>"$LOG"
import json, sys
tag = sys.argv[1]
rows = []
for ln in open("results/ce_ft/%s/train_log.jsonl" % tag, encoding="utf-8"):
    r = json.loads(ln)
    if r.get("event") in ("epoch0_zeroshot", "epoch", "best", "done"):
        rows.append(r)
ep = [r for r in rows if r.get("event") == "epoch"]
print("[summary] %s: e0=%.4f " % (tag, rows[0].get("val_ndcg10", 0)) +
      " ".join("e%d=%.4f/%.4f" % (r["epoch"], r["val_ndcg10"], r["val_ndcg10_domain_macro"])
               for r in ep) +
      " best=%.4f" % [r for r in rows if r.get("event") == "best"][-1].get("score", 0))
PY
  log "DONE $tag"
}

run_arm lx_possum --loss possum
run_arm lx_allpos --loss allpos
run_arm lx_ndcg   --loss ndcg
run_arm lx_sqrt   --loss possum --sampler sqrt

log "=== loss experiment done"
