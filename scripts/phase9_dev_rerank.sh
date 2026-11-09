#!/usr/bin/env bash
# Re-rank the dev candidates with a fine-tuned checkpoint, all 13 domains.
# Same preprocessing as the zero-shot runs: title query (256), 250-token
# windows, stride 192, max-pool.
#   CKPT=results/ce_ft/<tag>/best TAG=sftv1 bash scripts/phase9_dev_rerank.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT=${CKPT:?pass CKPT=path/to/checkpoint}
TAG=${TAG:?pass TAG=name}
LOG=logs/phase9_dev_rerank_${TAG}.log
exec >>"$LOG" 2>&1
echo "=== SFT dev rerank tag=$TAG ckpt=$CKPT $(date +%H:%M:%S)"
for d in bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace; do
  out=runs/rerank_${TAG}_${d}_dev.trec
  [ -s "$out" ] && { echo "skip $d"; continue; }
  $PY scripts/run_chunk_rerank.py --domain "$d" --split dev \
    --run runs/dense_mxbai_${d}_dev.trec --k 100 \
    --model "$CKPT" --q-mode title --q-cap 256 --win 250 --stride 192 \
    --pool max --batch 32 --out "$out" \
    --meta results/rerank_${TAG}_${d}_dev.meta.json || echo "FAIL $d"
done
echo "PHASE9 DEV RERANK DONE $(date +%H:%M:%S)"
