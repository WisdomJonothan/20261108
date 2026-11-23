#!/bin/bash
cd /homeC/wujiaheng/RETECO || exit 1
for TAG in nohard pu10; do
  CK=/tmp/reteco/ce_ft/${TAG}_ce/best
  if [ -f "$CK/config.json" ]; then echo "SKIP train $TAG"; else
    echo "=== train $TAG $(date +%H:%M) ==="
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/train_ce.py --model models/bge-reranker-v2-m3 \
      --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
      --groups-prefix results/ce_folds/${TAG}/g_${TAG} --run-tag mxbai --val-frac 0.12 --val-queries 4 \
      --val-depth 100 --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
      --epochs 2 --lr 1.5e-5 --seed 17 --tag ${TAG}_ce --outdir /tmp/reteco/ce_ft/${TAG} > logs/${TAG}_ce.log 2>&1
  fi
  echo "=== eval $TAG $(date +%H:%M) ==="
  bash ndcg_dev_chain.sh 0 ${TAG} 2>&1 | tail -2
done
echo PU_CHAIN_DONE
