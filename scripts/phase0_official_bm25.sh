#!/usr/bin/env bash
# Reference check: run the official starter-kit BM25 (gensim LuceneBM25Model)
# so we can compare our scoring code against the published baseline numbers.
set -u
export JAVA_HOME='D:\RETECO\env\jdk-21.0.13+9'
export PATH="/d/RETECO/env/jdk-21.0.13+9/bin:$PATH"
export PYTHONIOENCODING=utf-8

PY=/d/anaconda/python.exe
ROOT=/d/RETECO
cd $ROOT/scripts || exit 1

for dom in iota bitcoin; do
  for split in dev train; do
    echo "===== official BM25 $dom/$split"
    $PY run_bm25_official.py \
      --data "$ROOT/data/track1_tempo/$dom" --split $split \
      --out "$ROOT/runs/bm25_official_${dom}_${split}.trec" \
      --meta "$ROOT/results/bm25_official_${dom}_${split}.meta.json" \
      2>&1 | tee "$ROOT/logs/bm25_official_${dom}_${split}.log"
    $PY score_run.py --run "$ROOT/runs/bm25_official_${dom}_${split}.trec" \
      --data "$ROOT/data/track1_tempo/$dom" --split $split \
      --json "$ROOT/results/score_bm25_official_${dom}_${split}.json" \
      --tag "official-bm25 $dom $split" 2>&1 | tee -a "$ROOT/logs/bm25_official_${dom}_${split}.log"
  done
done
echo "OFFICIAL DONE"
