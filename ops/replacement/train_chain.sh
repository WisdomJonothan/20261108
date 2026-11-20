#!/bin/bash
cd ~/RETECO
M=/tmp/reteco/ce_ft
mkdir -p $M logs
# wait for the fold-0 run launched earlier to finish
while pgrep -f "train_ce.py.*--outdir $M/f0" > /dev/null; do sleep 20; done
touch $M/f0.trained; echo "[chain] f0 finished $(date)"
for F in f1 f2 fall; do
  [ -f $M/$F.trained ] && continue
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/train_ce.py \
    --model models/bge-reranker-v2-m3 \
    --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
    --groups-prefix results/ce_folds/$F/g_$F \
    --run-tag mxbai --val-frac 0.12 --val-queries 4 --val-depth 100 \
    --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
    --epochs 2 --lr 1.5e-5 --seed 17 --tag ${F}_ce --outdir $M/$F > logs/${F}_ce.log 2>&1
  touch $M/$F.trained; echo "[chain] $F finished $(date)"
done
echo "[chain] ALL_DONE $(date)"
