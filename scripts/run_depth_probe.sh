#!/usr/bin/env bash
# Depth probe: rerank a 200-deep pool for the 62 headroom queries selected by
#   python scripts/depth_probe.py --select
# and then compare fused nDCG@10 at pool 100 vs pool 200.
#   CKPT=results/ce_ft/v2m3_pool8_e1/best TAG=depth bash scripts/run_depth_probe.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
CKPT=${CKPT:?pass CKPT=path/to/checkpoint}
TAG=${TAG:-depth}
LOG=logs/depth_probe_${TAG}.log
exec >>"$LOG" 2>&1
echo "=== depth probe tag=$TAG ckpt=$CKPT $(date +%H:%M:%S)"
for d in bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace; do
  ex=results/depth_probe/examples_depth_${d}.jsonl
  [ -s "$ex" ] || { echo "no sample $d"; continue; }
  out=runs/rerank_depth_${d}_dev.trec
  [ -s "$out" ] && { echo "skip $d"; continue; }
  $PY scripts/run_chunk_rerank.py --domain "$d" --split dev \
    --examples "$ex" --run runs/dense_mxbai_${d}_dev.trec --k 200 \
    --model "$CKPT" --q-mode title --q-cap 256 --win 250 --stride 192 \
    --pool max --batch 32 --out "$out" \
    --meta results/depth_probe/rerank_${d}.meta.json || echo "FAIL $d"
done
echo "DEPTH PROBE RERANK DONE $(date +%H:%M:%S)"
$PY scripts/depth_probe.py --score
echo "DEPTH PROBE SCORING DONE $(date +%H:%M:%S)"
