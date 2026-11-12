#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
ROOT=${RETEco_ROOT:-$(pwd)}
MODELS=${RETEco_MODELS:-$ROOT/models}
PY=/d/anaconda/python.exe; export PYTHONUTF8=1
LOG=logs/phase8_mxbai_rerank.log
exec >>"$LOG" 2>&1
for d in bitcoin economics politics genealogy hsm history law workplace cardano monero quant iota; do
  out="runs/rerank_mxbai_${d}_dev.trec"
  [ -s "$out" ] && { echo "skip $d"; continue; }
  echo "=== rerank mxbai $d $(date +%H:%M:%S)"
  $PY scripts/run_chunk_rerank.py --domain "$d" --split dev \
     --run "runs/dense_mxbai_${d}_dev.trec" --k 100 \
     --model ${MODELS:-$ROOT/models}/bge-reranker-v2-m3 \
     --q-mode title --q-cap 256 --win 250 --pool max --batch 32 \
     --out "$out" --meta "results/rerank_mxbai_${d}_dev.meta.json" \
     || echo "FAIL rerank $d"
done
echo "RERANK DONE $(date +%H:%M:%S)"
