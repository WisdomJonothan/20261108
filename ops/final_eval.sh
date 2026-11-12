#!/usr/bin/env bash
# Final evaluation of one candidate on the official dev split, end to end.
#
#   bash ops/final_eval.sh <TAG> <CKPT> <Q-MODE> <Q-CAP-WORDS> [WA] [WB]
#
# <Q-MODE> is one of title|body|full|ctxt|summ; the constraint-preserving forms
# go through run_chunk_rerank_qvar.py so that dev sees exactly the query text
# the train-internal validation used.  Re-ranking is resumable: an existing
# runs/rerank_<TAG>_<domain>_dev.trec is skipped, so an interrupted run can be
# restarted with the same command.
#
# Produces, by delegating to the canonical scripts:
#   runs/rerank_<TAG>_<d>_dev.trec -> runs/fused_<TAG>_<d>_dev.trec
#   -> runs/sub_<TAG>_<d>_dev.trec (de-tied, validated)
#   -> results/score_sub_<TAG>_<d>_dev.json
#   -> results/aggregate_sub_<TAG>_dev.json                (domain-macro)
#   -> results/boot_dom_<TAG>_vs_zs.json                   (paired bootstrap)
set -u
cd /d/RETECO || exit 1
TAG=${1:?usage: final_eval.sh TAG CKPT QMODE QCAP [WA] [WB]}
CKPT=${2:?missing CKPT}
QMODE=${3:-title}
QCAP=${4:-256}
WA=${5:-1}
WB=${6:-1}

PY=/d/anaconda/python.exe
PS=powershell.exe
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
LOG=logs/final_$TAG.log
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -d "$CKPT" ]; then echo "no checkpoint at $CKPT"; exit 1; fi

gpu_free() {
  local first
  first=$($PS -NoProfile -ExecutionPolicy Bypass -File ops/gpu_busy.ps1 2>/dev/null | head -1 | cut -d' ' -f1)
  [ "$first" = "FREE" ]
}
wait_gpu() {
  local cap=${1:-240} need=${2:-2} ok=0 waited=0
  while [ $ok -lt $need ] && [ $waited -lt $cap ]; do
    if gpu_free; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge $need ] && return 0
    sleep 60; waited=$((waited+1))
  done
  return 1
}

log "start tag=$TAG ckpt=$CKPT qmode=$QMODE qcap=$QCAP fusion z $WA:$WB"
for d in $DOMAINS; do
  out=runs/rerank_${TAG}_${d}_dev.trec
  if [ -s "$out" ]; then log "skip $d (exists)"; continue; fi
  if ! wait_gpu 180 2; then log "GPU busy before $d; stopping (resume with the same command)"; exit 1; fi
  case "$QMODE" in
    title|body|full|ctxt|summ)
      $PY -u scripts/run_chunk_rerank_qvar.py --domain "$d" --split dev \
        --run "runs/dense_mxbai_${d}_dev.trec" --k 100 --model "$CKPT" \
        --q-mode "$QMODE" --q-cap-words "$QCAP" --q-tokens 256 \
        --win 250 --stride 192 --pool max --batch 32 \
        --out "$out" --meta "runs/rerank_${TAG}_${d}_dev.meta.json" \
        >> "logs/final_${TAG}_$d.log" 2>&1 ;;
    *)
      log "unknown q-mode $QMODE"; exit 1 ;;
  esac
  if [ $? -ne 0 ] || [ ! -s "$out" ]; then log "FAIL rerank $d"; exit 1; fi
  log "reranked $d"
done

log "fusing ($WA:$WB) and scoring"
for d in $DOMAINS; do
  $PY scripts/fuse_scores.py --run-a "runs/dense_mxbai_${d}_dev.trec"     --run-b "runs/rerank_${TAG}_${d}_dev.trec" --norm z --wa "$WA" --wb "$WB"     --out "runs/fused_${TAG}_${d}_dev.trec"     --json "results/fuse_${TAG}_${d}_dev.json" >> "$LOG" 2>&1 || { log "FAIL fuse $d"; exit 1; }
done
$PY scripts/validate_submission.py --runs "runs/fused_${TAG}_*_dev.trec" --split dev   --detie "runs/sub_${TAG}_{domain}_dev.trec"   --json "results/submission_validation_${TAG}.json" >> "$LOG" 2>&1 || { log "FAIL validate"; exit 1; }
for d in $DOMAINS; do
  $PY scripts/score_run.py --run "runs/sub_${TAG}_${d}_dev.trec"     --data "data/track1_tempo/${d}" --split dev     --json "results/score_sub_${TAG}_${d}_dev.json" >> "$LOG" 2>&1 || { log "FAIL score $d"; exit 1; }
done
$PY scripts/aggregate_scores.py --pattern "results/score_sub_${TAG}_*_dev.json"   --tag "$TAG" --json "results/aggregate_sub_${TAG}_dev.json" >> "$LOG" 2>&1 || { log "FAIL aggregate"; exit 1; }

log "bootstrap vs zero-shot"
$PY scripts/bootstrap_compare.py --a results/score_sub_zs11_*_dev.json \
  --b results/score_sub_${TAG}_*_dev.json --stat domain \
  --label "$TAG vs zs11" --json "results/boot_dom_${TAG}_vs_zs.json" >> "$LOG" 2>&1
log "bootstrap vs frozen arm (sft_e2bc)"
$PY scripts/bootstrap_compare.py --a results/score_sub_sft_e2bc_*_dev.json \
  --b results/score_sub_${TAG}_*_dev.json --stat domain \
  --label "$TAG vs frozen" --json "results/boot_dom_${TAG}_vs_frozen.json" >> "$LOG" 2>&1
log "multi-evidence"
$PY scripts/dev_multievidence.py --runs "runs/sub_${TAG}_{domain}_dev.trec" \
  --out "results/MULTIEVIDENCE_${TAG}.json" >> "$LOG" 2>&1 || log "WARN multievidence"
tail -3 "$LOG"
log "=== final eval done: $TAG"
