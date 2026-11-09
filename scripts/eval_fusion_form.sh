#!/usr/bin/env bash
# Fuse dense mxbai with a CE rerank arm at a given fusion form, score it on dev
# for all 13 domains, and write the per-domain JSONs plus an aggregate.
#
#   TAG=sft_e1b FORM=rrf WA=1 WB=4 bash scripts/eval_fusion_form.sh
#   TAG=mxbai   FORM=rrf WA=1 WB=4 bash scripts/eval_fusion_form.sh
#
# Only RRF fusion is supported here (z-sum arms come from freeze_sft_arm.sh).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
TAG=${TAG:?pass TAG=sft_e1b|mxbai}
FORM=${FORM:-rrf}
WA=${WA:-1}; WB=${WB:-1}
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
NAME="${TAG}_${FORM}_${WA}${WB}"
LOG=logs/eval_fusion_${NAME}.log
exec >>"$LOG" 2>&1
echo "=== eval start $NAME $(date +%H:%M:%S)"

if [ "$TAG" = "mxbai" ]; then
  RERANK_PAT="runs/rerank_mxbai_%s_dev.trec"
else
  RERANK_PAT="runs/rerank_${TAG}_%s_dev.trec"
fi

for d in $DOMAINS; do
  ra="runs/dense_mxbai_${d}_dev.trec"
  rb=$(printf "$RERANK_PAT" "$d")
  if [ ! -s "$rb" ]; then echo "skip $d (no rerank)"; continue; fi
  out="runs/fused_${NAME}_${d}_dev.trec"
  if [ "$FORM" = "rrf" ]; then
    $PY scripts/fuse_rrf.py --run-a "$ra" --run-b "$rb" --wa "$WA" --wb "$WB" --c 60 \
      --out "$out" --json "results/fuse_${NAME}_${d}_dev.json" >/dev/null || { echo "FAIL fuse $d"; continue; }
  else
    $PY scripts/fuse_zsum.py --run-a "$ra" --run-b "$rb" --wa "$WA" --wb "$WB" \
      --out "$out" --json "results/fuse_${NAME}_${d}_dev.json" >/dev/null || { echo "FAIL fuse $d"; continue; }
  fi
  $PY scripts/score_run.py --run "$out" --data "data/track1_tempo/$d" --split dev \
    --json "results/score_${NAME}_${d}_dev.json" >/dev/null || echo "FAIL score $d"
done

$PY scripts/aggregate_scores.py --pattern "results/score_${NAME}_*_dev.json" \
  --tag "$NAME" --json "results/aggregate_${NAME}_dev.json"
echo "=== eval done $(date +%H:%M:%S)"
