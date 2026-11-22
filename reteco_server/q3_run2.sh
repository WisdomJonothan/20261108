cd /homeC/wujiaheng/RETECO
ls runs/ | grep -i "frozenrepro\|sub_sft" | head -5
for d in bitcoin quant law; do
  .venv/bin/python scripts/score_run.py --run runs/q3ce_${d}_dev.trec --data data/track1_tempo/$d --split dev --json results/q3ce/score_${d}.json >/dev/null 2>&1
  .venv/bin/python scripts/score_run.py --run runs/sub_sft_e2bc_${d}_dev.trec --data data/track1_tempo/$d --split dev --json results/q3ce/fr_${d}.json >/dev/null 2>&1
done
.venv/bin/python - <<'PY'
import json
for d in ['bitcoin','quant','law']:
    a=json.load(open(f'results/q3ce/score_{d}.json'))['nDCG@10']
    b=json.load(open(f'results/q3ce/fr_{d}.json'))['nDCG@10']
    print(f'{d:8s} qwen3pool fused {a:.4f}  frozen {b:.4f}  delta {a-b:+.4f}')
PY
