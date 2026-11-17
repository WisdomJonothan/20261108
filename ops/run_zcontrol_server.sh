#!/usr/bin/env bash
# Run from the project tree; select the existing server environment with PYTHON_BIN.
# Deliberately preserves the original accumulation, windows and selection protocol.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RETEco_ROOT="$ROOT"
exec "${PYTHON_BIN:-python}" -u scripts/train_ce_zcontrol.py \
  --model models/bge-reranker-v2-m3 \
  --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
  --groups-prefix results/fusion100_groups --run-tag mxbai \
  --tag fusion100_full_zce_s17_20261116 --outdir results/ce_ft \
  --z-ce-only --fusion-select --loss ndcg --val-metric domain_macro \
  --epochs 2 --lr 1.5e-5 --warmup-frac 0.1 --weight-decay 0.01 --grad-accum 4 \
  --q-mode full --q-cap-words 0 --q-cap 256 --win 250 --stride 192 \
  --window-mode max --max-windows 3 --max-len 508 --pool-chunk 4 \
  --dtype bf16 --grad-ckpt --freeze-layers 8 --seed 17 \
  --val-qids results/fusion100_split.json --val-depth 100 --dump-val-scores \
  --val-frac 0.12 --val-queries 12 --eval-win-limit 0 --keep-last 1 \
  --sampler uniform --step-targets results/ce_step_targets --step-lambda 0
