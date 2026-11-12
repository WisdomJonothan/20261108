#!/usr/bin/env bash
# Score the retrained SFT CE (e1b) run on its own, for all 13 domains.
set -u
cd /d/RETECO
PY=/d/anaconda/python.exe; export PYTHONUTF8=1
TAG=${TAG:-sft_e1b}
LOG=logs/score_${TAG}_dev.log
exec >>"$LOG" 2>&1
for d in bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace; do
  out=results/score_${TAG}_${d}_dev.json
  [ -s "$out" ] && continue
  $PY scripts/score_run.py --run "runs/rerank_${TAG}_${d}_dev.trec" \
    --data "data/track1_tempo/$d" --split dev --json "$out" >/dev/null || echo "FAIL $d"
done
$PY scripts/aggregate_scores.py --pattern "results/score_${TAG}_*_dev.json" --tag "$TAG" \
  --json "results/aggregate_${TAG}_dev.json"
echo "DONE $(date +%H:%M:%S)"
