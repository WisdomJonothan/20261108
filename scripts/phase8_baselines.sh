#!/usr/bin/env bash
# Phase 8: strong zero-shot retrievers on the same dev queries (RETECO Track 1a).
# gte-large-en-v1.5 skipped: custom RoPE/pack_qkv code incompatible with transformers 5.x.
set -u
cd "$(dirname "$0")/.."
ROOT=${RETEco_ROOT:-$(pwd)}
MODELS=${RETEco_MODELS:-$ROOT/models}
PY=/d/anaconda/python.exe
export PYTHONUTF8=1
DATA=data/track1_tempo
LOG=logs/phase8_baselines.log
exec >>"$LOG" 2>&1

run_model () {
  local tag=$1 model=$2 qpfx=$3 dpfx=$4
  shift 4
  echo "########## $tag $(date +%H:%M:%S)"
  for d in "$@"; do
    out="runs/dense_${tag}_${d}_dev.trec"
    [ -s "$out" ] && { echo "skip $d"; continue; }
    echo "=== $tag $d $(date +%H:%M:%S)"
    $PY scripts/run_dense_generic.py --index-data "$DATA/$d" --split dev \
        --model "$model" --tag "$tag" --pooling mean \
        --query-prefix "$qpfx" --doc-prefix "$dpfx" \
        --work emb --out "$out" --meta "runs/dense_${tag}_${d}_dev.meta.json" \
        || { echo "FAIL $tag $d"; continue; }
    $PY scripts/score_run.py --run "$out" --data "$DATA/$d" --split dev \
        --json "results/${tag}_${d}_dev.json" || echo "FAIL score $tag $d"
  done
}

# e5-base-v2 first: fastest, and it is the family the TEMPO paper compares against.
run_model e5base ${MODELS:-$ROOT/models}/e5-base-v2 "query: " "passage: " \
  iota quant law workplace cardano monero bitcoin economics travel genealogy politics hsm history
# mxbai only on the three domains that carry most of the argument (335M is ~5x slower).
run_model mxbai ${MODELS:-$ROOT/models}/mxbai-embed-large-v1 \
  "Represent this sentence for searching relevant passages: " "" \
  bitcoin economics politics
echo "ALL DONE $(date +%H:%M:%S)"
