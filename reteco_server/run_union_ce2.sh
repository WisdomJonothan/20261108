#!/usr/bin/env bash
set -u
cd "$HOME/RETECO" || exit 1
for d in bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace; do
  qids=$(cut -d' ' -f1 runs/union_zero_dev.trec | sort -u | grep -f <(ls data/track1_tempo/$d/examples_dev.jsonl >/dev/null; .venv/bin/python - "$d" <<'PY'
import json,sys
d=sys.argv[1]
for line in open(f"data/track1_tempo/{d}/examples_dev.jsonl", encoding="utf-8"):
    print(json.loads(line)["id"])
PY
) || true)
  if [ -z "$qids" ]; then continue; fi
  awk -v ids="$qids" 'BEGIN{n=split(ids,a," "); for(i=1;i<=n;i++) keep[a[i]]=1} keep[$1]' runs/union_zero_dev.trec > runs/union_zero_${d}.trec
  [ -s runs/union_zero_${d}.trec ] || continue
  echo "=== $d $(wc -l < runs/union_zero_${d}.trec) lines ==="
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_chunk_rerank_qvar.py --domain $d --split dev \
    --run runs/union_zero_${d}.trec --k 200 --q-mode title --q-cap-words 0 --pair-max-len 510 \
    --q-tokens 256 --win 250 --stride 192 --pool max --model results/ce_ft/v2m3_pool8_e2/best \
    --out runs/union_zero_${d}_ce.trec --meta /tmp/reteco/uz_${d}.meta.json --batch 16 2>&1 | tail -2
done
echo UNION_CE_DONE
