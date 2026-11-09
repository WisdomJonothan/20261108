#!/usr/bin/env bash
# After phase11_sft_oof.sh has reranked dev with the OOF retrained checkpoint,
# score the two fusion forms that matter and print the comparison the paper
# needs: z 1:1 (the frozen published form) and RRF 1:1 (what the out-of-fold
# fit selected).
#
#   TAG=sft_e1b bash scripts/phase11_finish_eval.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
TAG=${TAG:?pass TAG=name of the rerank runs}
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
LOG=logs/phase11_finish_eval_${TAG}.log
exec >>"$LOG" 2>&1
echo "=== eval start tag=$TAG $(date +%H:%M:%S)"

# z 1:1 (frozen form): fuse -> de-tie -> official validation -> score -> aggregate
TAG="$TAG" bash scripts/freeze_sft_arm.sh || echo "FAIL freeze z11"

# RRF 1:1 (OOF-selected form)
for d in $DOMAINS; do
  $PY scripts/fuse_rrf.py --run-a "runs/dense_mxbai_${d}_dev.trec" \
    --run-b "runs/rerank_${TAG}_${d}_dev.trec" --wa 1 --wb 1 --c 60 \
    --out "runs/fused_${TAG}_rrf11_${d}_dev.trec" \
    --json "results/fuse_${TAG}_rrf11_${d}_dev.json" >/dev/null \
    && $PY scripts/score_run.py --run "runs/fused_${TAG}_rrf11_${d}_dev.trec" \
       --data "data/track1_tempo/$d" --split dev \
       --json "results/score_${TAG}_rrf11_${d}_dev.json" >/dev/null \
    || echo "FAIL rrf $d"
done
$PY scripts/aggregate_scores.py --pattern "results/score_${TAG}_rrf11_*_dev.json" \
  --tag "${TAG}_rrf11" --json "results/aggregate_${TAG}_rrf11_dev.json"

# the two reference arms, re-aggregated so the table is self-contained
for ref in sub_v1 sub_sft11; do
  $PY scripts/aggregate_scores.py --pattern "results/score_${ref}_*_dev.json" \
    --tag "$ref" --json "results/aggregate_${ref}_dev.json" 2>/dev/null || true
done
echo "=== eval done $(date +%H:%M:%S)"
