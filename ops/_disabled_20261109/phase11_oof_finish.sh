#!/usr/bin/env bash
# Finish the out-of-fold fusion fit that phase11_sft_oof.sh left half-done:
# subsets were written by a buggy writer, so rerun subset -> OOF rerank -> fit.
set -u
cd /d/RETECO
PY=/d/anaconda/python.exe; export PYTHONUTF8=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TAG=${TAG:-v2m3_pool8_e1b}
LOG=logs/phase11_oof_finish_${TAG}.log
exec >>"$LOG" 2>&1
echo "=== oof finish start $(date +%H:%M:%S)"
DOMS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
for d in $DOMS; do
  ex=results/ce_val45_e1b/examples_${d}.jsonl
  sub=runs/subset_e1b_${d}_train.trec
  out=runs/rerank_e1b_${d}_oof.trec
  $PY scripts/make_subset_run.py --run runs/dense_mxbai_${d}_train.trec \
      --examples "$ex" --out "$sub" >/dev/null || { echo "FAIL subset $d"; continue; }
  [ -s "$out" ] && { echo "skip $d"; continue; }
  $PY scripts/run_chunk_rerank.py --domain "$d" --split oof \
    --examples "$ex" --run "$sub" --k 100 \
    --model "results/ce_ft/$TAG/best" --q-mode title --q-cap 256 --win 250 --stride 192 \
    --pool max --batch 32 --out "$out" \
    --meta results/rerank_e1b_${d}_oof.meta.json || echo "FAIL oof $d"
done
echo "--- rerank done $(date +%H:%M:%S)"
$PY scripts/fit_fusion_weight.py --split train \
  --dense-pattern "runs/subset_e1b_{domain}_train.trec" \
  --ce-pattern "runs/rerank_e1b_{domain}_oof.trec" \
  --select-metric domain_macro --out results/fusion_fit_oof_e1b.json
echo "--- fit done $(date +%H:%M:%S)"
