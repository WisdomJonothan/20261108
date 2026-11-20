#!/bin/bash
cd ~/RETECO
for K in 1 2; do
  [ -s results/replacement_v2/exp_budget_cap$K.json ] && continue
  echo "[budget] cap$K $(date)"
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/benchmark_pairwise_replacement_v2.py \
    --root ~/RETECO --candidates results/replacement_v2/candidates_dev_cap$K.json \
    --policies direct --batch-size 16 \
    --out results/replacement_v2/exp_budget_cap$K.json > logs/budget_cap$K.log 2>&1
  echo "[budget] cap$K done rc=$? $(date)"
done
