#!/bin/bash
# usage: gate_ce_infer.sh <fold: f0|f1|f2|fall>
# For one fold: score its held-out (or calibration) queries with the fold-only CE,
# z-fuse that CE with dense (1:1) into a "frozen-analog" ranking, then build the
# frozen candidate/pair list for those queries.
set -e
cd ~/RETECO
F=$1
case "$F" in
  f0|f1|f2) FOLDKW=$F ;;
  fall) FOLDKW=fall ;;
  *) echo "bad fold $F"; exit 2 ;;
esac
DOMAINS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"
mkdir -p results/reteco results/replacement_v2/runs logs

# qid list for this model's held-out queries
python3 - "$F" <<'PY'
import json, sys
F = sys.argv[1]
m = json.load(open('results/replacement_v2/ce_folds.json'))
if F == 'fall':
    qids = [q for d, v in m['calibration'].items() for q in v]
else:
    qids = [q for d, v in m['folds'][F]['held'].items() for q in v]
json.dump(qids, open(f'results/replacement_v2/qids_{F}.json', 'w'))
print(F, 'qids', len(qids))
PY

MODEL=/tmp/reteco/ce_ft/$F/${F}_ce/best
if [ ! -f "$MODEL/config.json" ]; then
  echo "[warn] $F best/ absent (val never beat zero-shot); using the base model as best"
  MODEL=$PWD/models/bge-reranker-v2-m3
fi
for d in $DOMAINS; do
  if [ "$F" = "fall" ]; then
    EX=results/ce_folds/fall/ex_all_${d}.jsonl
  else
    EX=results/ce_folds/${F}/ex_${F}_${d}.jsonl
  fi
  [ -s "$EX" ] || { echo "[skip] $d no examples"; continue; }
  if [ -f "runs/${F}_ce_${d}_train_ce.trec" ] && [ -s "runs/${F}_ce_${d}_train_ce.trec" ]; then
    echo "[have] ce $d"
  else
    echo "[ce] $d"
    CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python scripts/run_chunk_rerank_qvar.py \
      --domain "$d" --examples "$EX" --split train \
      --run "runs/dense_mxbai_${d}_train.trec" --k 100 --q-mode title --q-cap-words 0 \
      --pair-max-len 510 --q-tokens 256 --win 250 --stride 192 --pool max \
      --model "$MODEL" \
      --out "runs/${F}_ce_${d}_train_ce.trec" \
      --meta "results/reteco/${F}_ce_${d}_train.meta.json" --batch 16 \
      >> "logs/${F}_infer.log" 2>&1
    tail -1 "logs/${F}_infer.log"
  fi
  if [ -f "runs/trainfuse_${F}_${d}_train.trec" ] && [ -s "runs/trainfuse_${F}_${d}_train.trec" ]; then
    echo "[have] fuse $d"
  else
    .venv/bin/python scripts/fusion_for_train.py --domain "$d" --ce-tag "${F}_ce" \
      --out-tag "trainfuse_$F" >> "logs/${F}_infer.log" 2>&1
    tail -1 "logs/${F}_infer.log"
  fi
done

echo "[candidates] $F"
.venv/bin/python scripts/build_replacement_candidates.py --split train \
  --frozen-tag "trainfuse_$F" --dense-tag dense_mxbai --ce-tag "${F}_ce" --ce-dir runs \
  --qids-file "results/replacement_v2/qids_$F.json" \
  --out-tag "candidates_train_$F" \
  > "logs/${F}_cand.log" 2>&1
tail -3 "logs/${F}_cand.log"
echo "[done] $F"
