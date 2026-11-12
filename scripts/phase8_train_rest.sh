#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}
for d in iota law quant cardano workplace travel; do
  dr=runs/dense_mxbai_${d}_train.trec
  if [ ! -s "$dr" ]; then
    echo "=== dense train $d $(date +%H:%M:%S)"
    $PY scripts/run_dense_generic.py --index-data data/track1_tempo/$d --split train \
      --model models/mxbai-embed-large-v1 --tag mxbai --pooling mean \
      --query-prefix "Represent this sentence for searching relevant passages: " \
      --work emb --out "$dr" --meta results/dense_mxbai_${d}_train.json
  fi
  rr=runs/rerank_mxbai_${d}_train.trec
  if [ ! -s "$rr" ]; then
    echo "=== ce train $d $(date +%H:%M:%S)"
    $PY scripts/run_chunk_rerank.py --domain $d --split train --run "$dr" --k 100 \
      --model models/bge-reranker-v2-m3 --q-mode title --q-cap 256 --win 250 --pool max \
      --batch 32 --out "$rr" --meta results/rerank_mxbai_${d}_train.meta.json
  fi
done
echo "TRAIN DONE $(date +%H:%M:%S)"
