#!/usr/bin/env bash
set -u
export TMP="D:\RETECO\tmp" TEMP="D:\RETECO\tmp"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
ROOT=/d/RETECO; PY=/d/anaconda/python.exe
LOG=$ROOT/logs/phase7_catchup.log; echo "START $(date '+%F %T')" > $LOG
for dom in workplace politics; do
  D=$ROOT/data/track1_tempo/$dom
  echo "=== $dom dense $(date '+%T')" >> $LOG
  $PY -u $ROOT/scripts/run_dense_bge.py --index-data $D --split dev \
      --model $ROOT/models/bge-base-en-v1.5 --work $ROOT/emb \
      --out $ROOT/runs/dense_bge_${dom}_dev.trec \
      --meta $ROOT/results/dense_bge_${dom}_dev.meta.json >> $LOG 2>&1 || { echo "dense FAIL $dom" >> $LOG; continue; }
  $PY $ROOT/scripts/score_run.py --run $ROOT/runs/dense_bge_${dom}_dev.trec --data $D --split dev \
      --json $ROOT/results/score_dense_bge_${dom}_dev.json --tag dense >> $LOG 2>&1
  echo "=== $dom rerank $(date '+%T')" >> $LOG
  $PY -u $ROOT/scripts/run_chunk_rerank.py --domain $dom --split dev \
      --run $ROOT/runs/dense_bge_${dom}_dev.trec --k 100 --q-cap 256 --q-mode title \
      --win 250 --stride 250 --pool max --model $ROOT/models/bge-reranker-v2-m3 \
      --out $ROOT/runs/rerank_v2m3_title_${dom}_dev.trec \
      --meta $ROOT/results/rerank_v2m3_title_${dom}_dev.meta.json >> $LOG 2>&1 || { echo "rerank FAIL $dom" >> $LOG; continue; }
  $PY $ROOT/scripts/score_run.py --run $ROOT/runs/rerank_v2m3_title_${dom}_dev.trec --data $D --split dev \
      --json $ROOT/results/score_rerank_v2m3_title_${dom}_dev.json --tag ce >> $LOG 2>&1
  echo "=== $dom fuse $(date '+%T')" >> $LOG
  $PY $ROOT/scripts/fuse_rrf.py --run-a $ROOT/runs/dense_bge_${dom}_dev.trec \
      --run-b $ROOT/runs/rerank_v2m3_title_${dom}_dev.trec \
      --out $ROOT/runs/twoview_${dom}_dev.trec --json $ROOT/results/twoview_${dom}_dev.json \
      --wa 2 --wb 1 --c 60 >> $LOG 2>&1
  $PY $ROOT/scripts/score_run.py --run $ROOT/runs/twoview_${dom}_dev.trec --data $D --split dev \
      --json $ROOT/results/score_twoview_${dom}_dev.json --tag fused >> $LOG 2>&1
  echo "=== $dom DONE $(date '+%T')" >> $LOG
done
echo "ALL DONE $(date '+%F %T')" >> $LOG
