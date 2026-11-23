#!/bin/bash
# Four-cell requirement experiment: one cell per GPU, three folds sequentially.
set -u
cd "$HOME/RETECO" || exit 1
mkdir -p logs results/replacement_v2/req
CELL="$1"
GPU="$2"
case "$CELL" in
  sel)  W=0 ;;
  req)  W=1 ;;
  both) W=2 ;;
  mis)  W=3 ;;
esac
for F in f0 f1 f2; do
  OUT="results/replacement_v2/req/${CELL}_${F}.json"
  if [ -f "$OUT" ]; then
    echo "skip $CELL $F"
    continue
  fi
  echo "=== $CELL $F start $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u scripts/req_cell.py \
    --root . --fold "$F" --cell "$CELL" --batch-size 8 \
    --out "$OUT" > "logs/req_${CELL}_${F}.log" 2>&1
  echo "=== $CELL $F exit $? $(date +%H:%M:%S)"
done
echo "REQ_CHAIN_DONE $CELL"
