#!/usr/bin/env bash
set -u
ROOT="${RETEco_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
for d in bitcoin economics politics genealogy hsm history monero; do
  out="results/temporal_pairs_${d}_train.jsonl"
  [ -s "$out" ] && { echo "skip $d"; continue; }
  echo "=== disc train $d $(date +%H:%M:%S)"
  PYTHONUTF8=1 /d/anaconda/python.exe scripts/temporal_discriminator.py \
    --domain "$d" --split train \
    --dense "runs/dense_mxbai_${d}_train.trec" \
    --ce "runs/rerank_mxbai_${d}_train.trec" \
    --k 100 --batch 64 --out "$out" 2>&1 | tail -3
done
echo "FEATURES DONE $(date +%H:%M:%S)"
