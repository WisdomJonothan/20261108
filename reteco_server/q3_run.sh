cd /homeC/wujiaheng/RETECO
mkdir -p results/q3ce
for d in quant law; do
  .venv/bin/python scripts/fuse_scores.py --run-a runs/dense_qwen3emb06_${d}_dev.trec \
    --run-b runs/q3ce_${d}_dev_ce.trec --out runs/q3ce_${d}_dev.trec \
    --json results/q3ce/fuse_${d}.json --norm z --wa 1 --wb 1 --tag q3ce >/dev/null 2>&1
done
for d in bitcoin quant law; do
  .venv/bin/python scripts/score_run.py --run runs/q3ce_${d}_dev.trec --data data/track1_tempo/$d --split dev --json results/q3ce/score_${d}.json >/dev/null 2>&1
  .venv/bin/python scripts/score_run.py --run runs/frozenrepro_${d}_dev.trec --data data/track1_tempo/$d --split dev --json results/q3ce/fr_${d}.json >/dev/null 2>&1
done
.venv/bin/python - <<'PY'
import json
tot=0
for d in ['bitcoin','quant','law']:
    a=json.load(open(f'results/q3ce/score_{d}.json'))['nDCG@10']
    b=json.load(open(f'results/q3ce/fr_{d}.json'))['nDCG@10']
    print(f'{d:8s} qwen3pool fused {a:.4f}  frozen(mxbai) fused {b:.4f}  delta {a-b:+.4f}')
PY
