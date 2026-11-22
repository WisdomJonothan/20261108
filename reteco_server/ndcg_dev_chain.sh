#!/usr/bin/env bash
# Evaluate the metric-aligned (lambdarank nDCG@10) CE on the official dev split.
# Mirrors the frozen recipe: title query, cap 256, window 250/192/max, pair-max-len 510,
# dense mxbai top-100 pool, z-normalised 1:1 fusion with the same dense run.
#
#   bash ndcg_dev_chain.sh [GPU]
set -u
cd "$HOME/RETECO" || exit 1
GPU=${1:-0}
TAG=${2:-ndcgdev}
CASE=${3:-ndcg}
if [ "$CASE" = "frozen" ]; then
  CKPT=results/ce_ft/v2m3_pool8_e2/best
else
  CKPT=/tmp/reteco/ce_ft/ndcg/ndcg_ce/best
fi
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

mkdir -p logs "results/$TAG"

for d in $DOMAINS; do
  CE=runs/${TAG}_${d}_dev_ce.trec
  if [ ! -s "$CE" ]; then
    .venv/bin/python -u scripts/run_chunk_rerank_qvar.py \
      --domain "$d" --split dev --run "runs/dense_mxbai_${d}_dev.trec" --k 100 \
      --q-mode title --q-cap-words 0 --pair-max-len 510 --q-tokens 256 \
      --win 250 --stride 192 --pool max --model "$CKPT" \
      --out "$CE" --meta "results/$TAG/${d}.meta.json" --batch 16 \
      >> "logs/${TAG}.log" 2>&1 || { echo "FAIL rerank $d"; exit 1; }
  fi
  FUS=runs/fused_${TAG}_${d}_dev.trec
  if [ ! -s "$FUS" ]; then
    .venv/bin/python -u scripts/fuse_scores.py \
      --run-a "runs/dense_mxbai_${d}_dev.trec" --run-b "$CE" \
      --norm z --wa 1 --wb 1 --out "$FUS" --json "results/$TAG/fuse_${d}.json" \
      >> "logs/${TAG}.log" 2>&1 || { echo "FAIL fuse $d"; exit 1; }
  fi
  .venv/bin/python -u scripts/score_run.py --run "$FUS" \
    --data "data/track1_tempo/${d}" --split dev \
    --json "results/$TAG/score_${d}.json" >> "logs/${TAG}.log" 2>&1 || { echo "FAIL score $d"; exit 1; }
  echo "done $d"
done

.venv/bin/python -u scripts/aggregate_scores.py \
  --pattern "results/$TAG/score_*_dev.json" --tag "$TAG" \
  --json "results/aggregate_${TAG}_dev.json" >> "logs/${TAG}.log" 2>&1
echo "NDCG_DEV_DONE $TAG"
