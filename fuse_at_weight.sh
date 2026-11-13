#!/usr/bin/env bash
# Fuse an existing frozen set of rerank runs at a given weight and score them.
#   bash ops/fuse_at_weight.sh <TAG> <RERANK_TAG> <WA> <WB> <OUT_TAG>
set -eu
cd "$(dirname "$0")/.."
PY=/d/anaconda/python.exe
TAG=$1; RTAG=$2; WA=$3; WB=$4; OUTTAG=$5
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
for d in $DOMAINS; do
  $PY scripts/fuse_scores.py --run-a "runs/dense_mxbai_${d}_dev.trec" \
    --run-b "runs/rerank_${RTAG}_${d}_dev.trec" --norm z --wa "$WA" --wb "$WB" \
    --out "runs/fused_${OUTTAG}_${d}_dev.trec" --json "results/fuse_${OUTTAG}_${d}_dev.json" >/dev/null
done
$PY scripts/validate_submission.py --runs "runs/fused_${OUTTAG}_*_dev.trec" --split dev \
  --detie "runs/sub_${OUTTAG}_{domain}_dev.trec" --json "results/submission_validation_${OUTTAG}.json" >/dev/null
for d in $DOMAINS; do
  $PY scripts/score_run.py --run "runs/sub_${OUTTAG}_${d}_dev.trec" --data "data/track1_tempo/${d}" \
    --split dev --json "results/score_sub_${OUTTAG}_${d}_dev.json" >/dev/null
done
$PY scripts/aggregate_scores.py --pattern "results/score_sub_${OUTTAG}_*_dev.json" --tag "$OUTTAG" \
  --json "results/aggregate_sub_${OUTTAG}_dev.json"
