#!/usr/bin/env bash
# Replicate the 2-epoch SFT arm under a different seed, end to end:
#   train (tag v2m3_pool8_<suffix>) -> dev rerank -> freeze (fuse/validate/score)
#   -> bootstrap vs the zero-shot z 1:1 arm.
#   SEED=23 SUFFIX=e2s23 bash scripts/phase12_replicate.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
SEED=${SEED:?pass SEED=23}
SUFFIX=${SUFFIX:?pass SUFFIX=e2s23}
TAG=v2m3_pool8_${SUFFIX}
OUT=results/ce_ft/$TAG

if [ ! -d "$OUT/best" ]; then
  EPOCHS=2 LR=1.5e-5 SEED="$SEED" TAG="$TAG" bash scripts/train_sft_variant.sh
fi
[ -d "$OUT/best" ] || { echo "no checkpoint in $OUT/best"; exit 1; }

CKPT=$OUT/best TAG="$SUFFIX" bash scripts/phase9_dev_rerank.sh
TAG="$SUFFIX" bash scripts/freeze_sft_arm.sh
$PY scripts/bootstrap_compare.py \
  --a results/score_sub_zs11_*_dev.json \
  --b results/score_sub_${SUFFIX}_*_dev.json \
  --label "${SUFFIX} vs zs z1:1" \
  --json results/boot_dom_${SUFFIX}_vs_zs.json --iters 10000 --seed 7 --stat domain
echo "REPLICATED $SUFFIX"
