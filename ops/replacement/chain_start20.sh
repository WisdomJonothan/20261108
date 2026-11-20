#!/bin/bash
cd $HOME/RETECO
while [ ! -f results/replacement_v2/reranker_start20_dev.json ]; do sleep 20; done
sleep 10
echo "[chain] reranker done $(date +%T)"
.venv/bin/python -u scripts/build_replacement_candidates.py --split dev \
  --frozen-tag start20 --dense-tag dense_mxbai --ce-tag fusion100_full_ce_s17_20261115 \
  --ce-dir runs --out-tag candidates_rr20_dev >> logs/start20_chain.log 2>&1
echo "[chain] candidates $(date +%T)" >> logs/start20_chain.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/benchmark_pairwise_replacement_v2.py --root $HOME/RETECO \
  --candidates results/replacement_v2/candidates_rr20_dev.json --policies direct --batch-size 16 \
  --out results/replacement_v2/exp_rr_start_dev.json > logs/rr_start.log 2>&1
echo "[chain] judge $(date +%T)" >> logs/start20_chain.log
