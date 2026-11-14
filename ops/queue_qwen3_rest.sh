#!/usr/bin/env bash
# Score all 13 dev domains with Qwen3-Reranker-0.6B, one python process per
# domain, model-card protocol (prefix + budgeted body + think suffix).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=/d/anaconda/python.exe
LOG=logs/qwen3_rerank3.log
echo "[$(date +%H:%M:%S)] driver start pid $$" >>"$LOG"
for d in bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace; do
  out="runs/rerank_qwen3_${d}_dev.trec"
  [ -s "$out" ] && [ -s "$out.meta.json" ] && { echo "[$(date +%H:%M:%S)] skip $d" >>"$LOG"; continue; }
  echo "[$(date +%H:%M:%S)] $d" >>"$LOG"
  $PY -u scripts/run_qwen3_rerank.py --domain "$d" \
      --run "runs/dense_mxbai_${d}_dev.trec" --out "$out" \
      --max-len 2048 --doc-cap 4000 --batch 16 >>"logs/qwen3_${d}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && echo "[$(date +%H:%M:%S)] FAIL $d rc=$rc" >>"$LOG"
done
echo "[$(date +%H:%M:%S)] driver done" >>"$LOG"
