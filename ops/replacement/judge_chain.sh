#!/bin/bash
cd ~/RETECO
for F in f0 f1 f2 fall; do
  while [ ! -s results/replacement_v2/candidates_train_$F.json ]; do sleep 30; done
  [ -s results/replacement_v2/exp_train_$F.json ] && continue
  echo "[judge] start $F $(date)"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/benchmark_pairwise_replacement_v2.py \
    --root ~/RETECO --candidates results/replacement_v2/candidates_train_$F.json \
    --policies direct --batch-size 16 --qrels-split train \
    --out results/replacement_v2/exp_train_$F.json > logs/judge_$F.log 2>&1
  echo "[judge] done $F $(date) rc=$?"
done
echo "[judge] ALL_DONE $(date)"
