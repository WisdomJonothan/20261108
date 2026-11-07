#!/usr/bin/env bash
# Phase 1: cross-encoder re-ranking on top of the dense (bge-base) candidate run.
set -uo pipefail
cd /d/RETECO
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
MODEL=/d/RETECO/models/bge-reranker-base
PY=/d/anaconda/python.exe

run_one () {
  local dom=$1 split=$2 k=$3
  local out="runs/rerank_bge_${dom}_${split}_k${k}.trec"
  local meta="results/rerank_bge_${dom}_${split}_k${k}.meta.json"
  local log="logs/rerank_bge_${dom}_${split}_k${k}.log"
  echo "=== rerank $dom/$split k=$k  $(date '+%H:%M:%S') ==="
  $PY scripts/run_rerank.py --domain "$dom" --split "$split" \
      --run "runs/dense_bge_${dom}_${split}.trec" --k "$k" \
      --model "$MODEL" --out "$out" --meta "$meta" 2>&1 | tee "$log" | tail -4
  $PY scripts/score_run.py --run "$out" --data "data/track1_tempo/$dom" \
      --split "$split" --json "results/score_rerank_bge_${dom}_${split}_k${k}.json" \
      --tag "rerank_bge_k${k}" 2>&1 | tail -3
}

for spec in "bitcoin dev 100" "bitcoin train 100" "bitcoin dev 50" "iota dev 100" "iota train 100"; do
  # shellcheck disable=SC2086
  run_one $spec
done
echo "ALL RERANK DONE $(date '+%H:%M:%S')"
