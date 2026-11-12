#!/usr/bin/env bash
# Final evaluation of one candidate on the official dev split, end to end.
#
#   bash ops/final_eval.sh <TAG> <CKPT> <Q-MODE> <Q-CAP-WORDS> [WA] [WB]
#
# <Q-MODE> is one of title|body|full|ctxt|summ; the constraint-preserving forms
# go through run_chunk_rerank_qvar.py so that dev sees exactly the query text
# the train-internal validation used.  Re-ranking is resumable, but only for the
# same (checkpoint contents, query mode, query cap, windowing, dense input) and a
# complete output: every step writes runs/*.done.json with a fingerprint plus the
# output's size and sha1, and a step is skipped only when all of that still
# matches.  ALLOW_ADOPT=1 stamps pre-existing untracked outputs instead of
# recomputing them (only when you know they came from this exact configuration).
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
ALLOW_ADOPT=${ALLOW_ADOPT:-0}

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
RERANK_PARAMS="qmode=$QMODE qcap=$QCAP qtokens=256 win=250 stride=192 pool=max batch=32"
for d in $DOMAINS; do
  out=runs/rerank_${TAG}_${d}_dev.trec
  dense=runs/dense_mxbai_${d}_dev.trec
  if $PY scripts/run_complete.py check --out "$out" --ckpt "$CKPT"        --params "$RERANK_PARAMS" --input "$dense" 2>/dev/null; then
    log "skip $d (complete: fingerprint match)"; continue
  fi
  if [ "$ALLOW_ADOPT" = "1" ] && [ -s "$out" ]; then
    log "WARN adopt $d: stamping an existing untracked rerank without recomputing it"
    $PY scripts/run_complete.py stamp --out "$out" --ckpt "$CKPT"       --params "$RERANK_PARAMS" --input "$dense" --note "adopted without recompute" || exit 1
    continue
  fi
  rm -f "$out.done.json"
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
  $PY scripts/run_complete.py stamp --out "$out" --ckpt "$CKPT"     --params "$RERANK_PARAMS" --input "$dense" || { log "FAIL stamp $d"; exit 1; }
  log "reranked $d"
done

log "fusing ($WA:$WB) and scoring"
for d in $DOMAINS; do
  $PY scripts/fuse_scores.py --run-a "runs/dense_mxbai_${d}_dev.trec"     --run-b "runs/rerank_${TAG}_${d}_dev.trec" --norm z --wa "$WA" --wb "$WB"     --out "runs/fused_${TAG}_${d}_dev.trec"     --json "results/fuse_${TAG}_${d}_dev.json" >> "$LOG" 2>&1 || { log "FAIL fuse $d"; exit 1; }
done
$PY scripts/validate_submission.py --runs "runs/fused_${TAG}_*_dev.trec" --split dev   --detie "runs/sub_${TAG}_{domain}_dev.trec"   --json "results/submission_validation_${TAG}.json" >> "$LOG" 2>&1 || { log "FAIL validate"; exit 1; }
for d in $DOMAINS; do
  $PY scripts/run_complete.py stamp --out "runs/sub_${TAG}_${d}_dev.trec" --ckpt "$CKPT"     --params "$RERANK_PARAMS fusion=z:$WA:$WB" --input "runs/fused_${TAG}_${d}_dev.trec"     || { log "FAIL stamp sub $d"; exit 1; }
done
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
