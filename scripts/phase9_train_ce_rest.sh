#!/usr/bin/env bash
# Zero-shot CE on the train split for the domains that still lack it, so the
# fusion weight (and the score-sum variant) can be fitted on TRAIN, not dev.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
LOG=logs/phase9_train_ce_rest.log
exec >>"$LOG" 2>&1
for d in cardano law quant travel workplace iota; do
  dr=runs/dense_mxbai_${d}_train.trec
  rr=runs/rerank_mxbai_${d}_train.trec
  [ -s "$dr" ] || { echo "skip $d (no dense train run)"; continue; }
  [ -s "$rr" ] && { echo "skip $d (already reranked)"; continue; }
  echo "=== ce train $d $(date +%H:%M:%S)"
  $PY scripts/run_chunk_rerank.py --domain "$d" --split train --run "$dr" --k 100 \
    --model models/bge-reranker-v2-m3 --q-mode title --q-cap 256 --win 250 --stride 192 \
    --pool max --batch 32 --out "$rr" --meta results/rerank_mxbai_${d}_train.meta.json \
    || echo "FAIL $d"
done
echo "PHASE9 TRAIN CE DONE $(date +%H:%M:%S)"
