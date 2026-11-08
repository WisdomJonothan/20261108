#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
LOG=logs/phase9_train_dense_new.log
exec >>"$LOG" 2>&1
for d in cardano law quant travel workplace; do
  dr=runs/dense_mxbai_${d}_train.trec
  [ -s "$dr" ] && { echo "skip $d"; continue; }
  echo "=== dense train $d $(date +%H:%M:%S)"
  $PY scripts/run_dense_generic.py --index-data data/track1_tempo/$d --split train \
    --model models/mxbai-embed-large-v1 --tag mxbai --pooling mean \
    --query-prefix "Represent this sentence for searching relevant passages: " \
    --work emb --out "$dr" --meta results/dense_mxbai_${d}_train.json || echo "FAIL $d"
done
echo "PHASE9 DENSE TRAIN DONE $(date +%H:%M:%S)"
