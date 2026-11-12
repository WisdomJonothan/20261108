#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}
for d in economics genealogy hsm bitcoin monero history; do
  out=results/temporal_pairs_${d}_dev.jsonl
  [ -s "$out" ] && { echo "skip $d"; continue; }
  echo "=== disc $d $(date +%H:%M:%S)"
  $PY scripts/temporal_discriminator.py --domain "$d" --split dev \
    --dense runs/dense_mxbai_${d}_dev.trec --ce runs/rerank_mxbai_${d}_dev.trec \
    --out "$out" >> logs/temporal_disc.log 2>&1 || echo "FAIL $d"
done
echo "DISC DONE $(date +%H:%M:%S)"
