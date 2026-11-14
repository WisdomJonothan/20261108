#!/usr/bin/env bash
# Re-run the two domains that hit OOM at batch 16: history (240 queries) and
# hsm (45), at batch 4 with expandable segments, then the four long queries at
# max-len 4096 and splice them in.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/d/anaconda/python.exe
LOG=logs/qwen3_missing.log
echo "[$(date +%H:%M:%S)] start" >>"$LOG"
for d in history hsm; do
  echo "[$(date +%H:%M:%S)] $d" >>"$LOG"
  $PY -u scripts/run_qwen3_rerank.py --domain "$d" --run "runs/dense_mxbai_${d}_dev.trec" \
     --out "runs/rerank_qwen3_${d}_dev.trec" --max-len 2048 --doc-cap 4000 --batch 4 \
     >>"logs/qwen3_${d}2.log" 2>&1
  echo "[$(date +%H:%M:%S)] $d rc=$?" >>"$LOG"
done
for spec in "hsm:18672_660" "history:70364_290,56254_754"; do
  d=${spec%%:*}; q=${spec#*:}
  $PY -u scripts/run_qwen3_rerank.py --domain "$d" --run "runs/dense_mxbai_${d}_dev.trec" \
     --out "/tmp/long_${d}.trec" --max-len 4096 --doc-cap 4000 --batch 4 --qids "$q" \
     >>"logs/qwen3_long_${d}.log" 2>&1
  $PY scripts/merge_trec_run.py --into "runs/rerank_qwen3_${d}_dev.trec" --from "/tmp/long_${d}.trec" --qids "$q" >>"$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] long $d done" >>"$LOG"
done
echo "[$(date +%H:%M:%S)] done" >>"$LOG"
