#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}
OUT=runs/dense_mxbai_travel_dev.trec
[ -s "$OUT" ] && exit 0
$PY scripts/run_dense_generic.py --index-data data/track1_tempo/travel --split dev \
  --model models/mxbai-embed-large-v1 --tag mxbai --pooling mean \
  --query-prefix "Represent this sentence for searching relevant passages: " \
  --work emb --batch-size 32 --out "$OUT" --meta results/dense_mxbai_travel_dev.json
