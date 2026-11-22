#!/bin/bash
cd /homeC/wujiaheng/RETECO || exit 1
for F in f0 f1 f2; do
  OUT=results/replacement_v2/expanded3_fixed_train_${F}.json
  [ -f "$OUT" ] && { echo "SKIP $F"; continue; }
  echo "=== $F $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/expand_actions3.py \
    --root . --split train --qids-file results/replacement_v2/qids_${F}.json \
    --start-tag trainfuse_${F} --ce-tag ${F}_ce \
    --max-actions 18 --budgets 2,6,12,18 --order prior --batch-size 8 \
    --out "$OUT" > logs/ea3_train_${F}.log 2>&1
  echo "done $F rc=$? $(date +%H:%M)"
done
echo EA3_TRAIN_CHAIN_DONE
