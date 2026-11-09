#!/usr/bin/env bash
# Freeze one SFT arm end to end: z 1:1 fusion -> de-tie + official validation
# -> score -> aggregate, all 13 domains.
#   TAG=sft_e2bc bash scripts/freeze_sft_arm.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
TAG=${TAG:?pass TAG=name of the rerank runs (runs/rerank_<TAG>_<d>_dev.trec)}
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
mkdir -p runs results
for d in $DOMAINS; do
  ce=runs/rerank_${TAG}_${d}_dev.trec
  [ -s "$ce" ] || { echo "MISSING $ce"; exit 1; }
  $PY scripts/fuse_scores.py --run-a runs/dense_mxbai_${d}_dev.trec --run-b "$ce" \
    --norm z --wa 1 --wb 1 --out runs/fused_${TAG}_${d}_dev.trec \
    --json results/fuse_${TAG}_${d}_dev.json >/dev/null
done
$PY scripts/validate_submission.py --runs "runs/fused_${TAG}_*_dev.trec" --split dev \
  --detie "runs/sub_${TAG}_{domain}_dev.trec" \
  --json results/submission_validation_${TAG}.json
for d in $DOMAINS; do
  $PY scripts/score_run.py --run runs/sub_${TAG}_${d}_dev.trec --data data/track1_tempo/$d \
    --split dev --json results/score_sub_${TAG}_${d}_dev.json >/dev/null
done
$PY scripts/aggregate_scores.py --pattern "results/score_sub_${TAG}_*_dev.json" \
  --tag "$TAG" --json results/aggregate_sub_${TAG}_dev.json
echo "FROZEN runs/sub_${TAG}_*_dev.trec"
