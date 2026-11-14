#!/usr/bin/env bash
# Score the Qwen3-Reranker baseline two ways: cross-encoder alone, and fused
# with the same dense scores at the frozen 1:1 z-weight.  Then aggregate.
set -eu
cd "$(dirname "$0")/.." || exit 1
PY=/d/anaconda/python.exe
for d in bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace; do
  ce="runs/rerank_qwen3_${d}_dev.trec"
  [ -s "$ce" ] || { echo "[skip] $d (no run)"; continue; }
  $PY scripts/fuse_scores.py --run-a "runs/dense_mxbai_${d}_dev.trec" --run-b "$ce" \
      --out "runs/sub_qwen3_${d}_dev.trec" --json "results/fuse_qwen3_${d}_dev.json" \
      --norm z --wa 1 --wb 1 --tag "qwen3_${d}" >/dev/null
  $PY scripts/score_run.py --run "$ce" --data "data/track1_tempo/${d}" --split dev \
      --json "results/score_ce_qwen3_${d}_dev.json" --tag "ce_qwen3_${d}" >/dev/null
  $PY scripts/score_run.py --run "runs/sub_qwen3_${d}_dev.trec" --data "data/track1_tempo/${d}" --split dev \
      --json "results/score_sub_qwen3_${d}_dev.json" --tag "sub_qwen3_${d}" >/dev/null
  echo "[ok] $d"
done
$PY scripts/aggregate_scores.py --pattern "results/score_ce_qwen3_*_dev.json" \
    --json results/aggregate_ce_qwen3_dev.json >/dev/null
$PY scripts/aggregate_scores.py --pattern "results/score_sub_qwen3_*_dev.json" \
    --json results/aggregate_sub_qwen3_dev.json >/dev/null
echo "[done] aggregates written"
