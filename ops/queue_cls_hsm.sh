#!/usr/bin/env bash
# Finish the CLS-pooling baseline: hsm alone (the domain that died on a 44.5 M-char
# document in the tokenizer), then aggregate all 13 domains.
set -u
cd /d/RETECO || exit 1
PY=/d/anaconda/python.exe
LOG=logs/cls_hsm.log
PREF="Represent this sentence for searching relevant passages: "
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "cls hsm start (char pre-truncation patch in run_dense_generic.py)"
$PY -u scripts/run_dense_generic.py --index-data data/track1_tempo/hsm --split dev \
  --model models/mxbai-embed-large-v1 --tag mxbai_cls --pooling cls \
  --query-prefix "$PREF" --topk 100 --work emb \
  --out runs/dense_mxbai_cls_hsm_dev.trec \
  --meta results/dense_mxbai_cls_hsm_dev.json >> "$LOG" 2>&1 || { log "FAIL dense hsm"; exit 1; }
log "encoded hsm"
$PY scripts/score_run.py --run runs/dense_mxbai_cls_hsm_dev.trec \
  --data data/track1_tempo/hsm --split dev \
  --json results/score_dense_mxbai_cls_hsm_dev.json >> "$LOG" 2>&1 || { log "FAIL score hsm"; exit 1; }
log "scored hsm"
$PY scripts/aggregate_scores.py --pattern "results/score_dense_mxbai_cls_*_dev.json" \
  --tag dense_mxbai_cls --json results/aggregate_dense_mxbai_cls_dev.json >> "$LOG" 2>&1 \
  || { log "FAIL aggregate"; exit 1; }
log "=== cls hsm + aggregate done"
