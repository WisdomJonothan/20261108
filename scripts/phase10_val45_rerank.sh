#!/usr/bin/env bash
# Out-of-fold fusion weight for the SFT cross-encoder (GPT review P0 #2).
#
# The dev split must not choose the fusion weight for the model we are about to
# publish, so rerank only the 45 held-out *train* queries that train_ce.py never
# saw, fit dense:CE on them, and report the dev number exactly once.
#
#   CKPT=results/ce_ft/v2m3_pool8_e1/best bash scripts/phase10_val45_rerank.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
CKPT=${CKPT:?pass CKPT=path/to/checkpoint}
TAG=${TAG:-val45}
LOG=logs/phase10_val45_${TAG}.log
exec >>"$LOG" 2>&1
echo "=== val45 rerank tag=$TAG ckpt=$CKPT $(date +%H:%M:%S)"

for d in bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace; do
  ex=results/ce_val45/examples_val45_${d}.jsonl
  sub=runs/subset_val45_${d}_train.trec
  out=runs/rerank_${TAG}_${d}_val45.trec
  [ -s "$out" ] && { echo "skip $d"; continue; }
  $PY scripts/make_subset_run.py --run runs/dense_mxbai_${d}_train.trec \
    --examples "$ex" --out "$sub" || { echo "FAIL subset $d"; continue; }
  $PY scripts/run_chunk_rerank.py --domain "$d" --split val45 \
    --examples "$ex" --run "$sub" --k 100 \
    --model "$CKPT" --q-mode title --q-cap 256 --win 250 --stride 192 \
    --pool max --batch 32 --out "$out" \
    --meta results/rerank_${TAG}_${d}_val45.meta.json || echo "FAIL $d"
done

$PY scripts/fit_fusion_weight.py --split train \
  --dense-pattern "runs/subset_val45_{domain}_train.trec" \
  --ce-pattern "runs/rerank_${TAG}_{domain}_val45.trec" \
  --select-metric domain_macro \
  --out results/fusion_fit_val45_${TAG}.json
echo "PHASE10 VAL45 DONE $(date +%H:%M:%S)"
