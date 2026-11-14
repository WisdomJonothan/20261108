#!/usr/bin/env bash
# Score all 13 dev domains with Qwen3-Reranker-0.6B, then aggregate.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/d/anaconda/python.exe
LOG=logs/qwen3_rerank.log
echo "[$(date +%H:%M:%S)] start" >>"$LOG"
for d in bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace; do
  [ -s "runs/rerank_qwen3_${d}_dev.trec" ] && { echo "[$(date +%H:%M:%S)] skip $d" >>"$LOG"; continue; }
  echo "[$(date +%H:%M:%S)] $d" >>"$LOG"
  $PY -u scripts/run_qwen3_rerank.py --domain "$d" \
      --run "runs/dense_mxbai_${d}_dev.trec" \
      --out "runs/rerank_qwen3_${d}_dev.trec" >>"$LOG" 2>&1 || { echo "[$(date +%H:%M:%S)] FAIL $d" >>"$LOG"; exit 1; }
done
echo "[$(date +%H:%M:%S)] done" >>"$LOG"
