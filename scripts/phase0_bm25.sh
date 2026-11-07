#!/usr/bin/env bash
# Phase-0 BM25: official prebuilt Pyserini index + official starter-kit BM25,
# for IOTA and Bitcoin, dev and train splits. Writes runs, metas, logs.
set -u
export JAVA_HOME='D:\RETECO\env\jdk-21.0.13+9'
export PATH="/d/RETECO/env/jdk-21.0.13+9/bin:$PATH"
export PYTHONIOENCODING=utf-8
export HF_ENDPOINT=https://hf-mirror.com

PY=/d/anaconda/python.exe
ROOT=/d/RETECO
mkdir -p $ROOT/runs $ROOT/results $ROOT/logs
cd $ROOT/scripts || exit 1

declare -A IDX=(
  [iota]="$ROOT/indexes/iota/lucene-inverted.reteco-v1.1-track1-iota.20261017.1.6.0"
  [bitcoin]="$ROOT/indexes/bitcoin/lucene-inverted.reteco-v1.1-track1-bitcoin.20261017.1.6.0"
)

for dom in iota bitcoin; do
  for split in dev train; do
    echo "===== pyserini BM25 $dom/$split"
    $PY run_bm25_pyserini.py \
      --index "${IDX[$dom]}" \
      --data "$ROOT/data/track1_tempo/$dom" \
      --split $split --k1 0.9 --b 0.4 --topk 1000 \
      --out "$ROOT/runs/bm25_pyserini_${dom}_${split}.trec" \
      --meta "$ROOT/results/bm25_pyserini_${dom}_${split}.meta.json" \
      2>&1 | tee "$ROOT/logs/bm25_pyserini_${dom}_${split}.log"
    $PY score_run.py --run "$ROOT/runs/bm25_pyserini_${dom}_${split}.trec" \
      --data "$ROOT/data/track1_tempo/$dom" --split $split \
      --json "$ROOT/results/score_bm25_pyserini_${dom}_${split}.json" \
      --tag "pyserini-bm25 $dom $split" 2>&1 | tee -a "$ROOT/logs/bm25_pyserini_${dom}_${split}.log"
  done
done
echo "ALL DONE"
