#!/usr/bin/env bash
# Post-training half of the seed replication (rerank -> freeze -> bootstrap).
# Separate from phase12_replicate.sh so the long rerank can be launched detached
# from the harness job tree (host kills bash jobs after ~10-11 min).
#   SUFFIX=e2s29 bash scripts/phase12_post.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
SUFFIX=${SUFFIX:?pass SUFFIX=e2s29}
OUT=results/ce_ft/v2m3_pool8_$SUFFIX
[ -x "$OUT/best" ] || [ -d "$OUT/best" ] || { echo "no checkpoint in $OUT/best"; exit 1; }
echo "[post $SUFFIX] start $(date +%H:%M:%S)"
CKPT=$OUT/best TAG="$SUFFIX" bash scripts/phase9_dev_rerank.sh
echo "[post $SUFFIX] rerank done $(date +%H:%M:%S)"
TAG="$SUFFIX" bash scripts/freeze_sft_arm.sh
echo "[post $SUFFIX] freeze done $(date +%H:%M:%S)"
$PY scripts/bootstrap_compare.py \
  --a results/score_sub_zs11_*_dev.json \
  --b results/score_sub_${SUFFIX}_*_dev.json \
  --label "${SUFFIX} vs zs z1:1" \
  --json results/boot_dom_${SUFFIX}_vs_zs.json --iters 10000 --seed 7 --stat domain
echo "[post $SUFFIX] ALL DONE $(date +%H:%M:%S)"
