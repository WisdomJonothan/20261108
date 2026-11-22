cd /homeC/wujiaheng/RETECO
mkdir -p results/q3ce
for d in bitcoin quant law; do
  .venv/bin/python scripts/fuse_scores.py --run-a runs/dense_qwen3emb06_${d}_dev.trec \
    --run-b runs/q3ce_${d}_dev_ce.trec --out runs/q3ce_${d}_dev.trec \
    --json results/q3ce/*.json --norm z --wa 1 --wb 1 --tag q3ce >/dev/null 2>&1
  .venv/bin/python scripts/score_run.py --run runs/q3ce_${d}_dev.trec --data data/track1_tempo/$d --split dev \
    --json results/q3ce/score_${d}.json > /dev/null 2>&1
  echo "$d $(.venv/bin/python -c "import json;d=json.load(open('results/q3ce/score_${d}.json'));print('fused %.4f dense %.4f'%(d['nDCG@10'], 0.0))")"
done
echo Q3_FUSE_DONE
