cd /homeC/wujiaheng/RETECO
for d in bitcoin quant law; do
  for i in $(seq 1 240); do
    [ -s runs/dense_qwen3emb06_${d}_dev.trec ] && break
    sleep 30
  done
  .venv/bin/python scripts/score_run.py --run runs/dense_qwen3emb06_${d}_dev.trec --data data/track1_tempo/$d --split dev --json /tmp/q3_$d.json >/dev/null 2>&1
  .venv/bin/python scripts/score_run.py --run runs/dense_mxbai_${d}_dev.trec --data data/track1_tempo/$d --split dev --json /tmp/mx_$d.json >/dev/null 2>&1
  .venv/bin/python - <<PY
import json
a=json.load(open('/tmp/q3_$d.json')); b=json.load(open('/tmp/mx_$d.json'))
print('$d', 'qwen3 nDCG@10 %.4f R@100 %.4f | mxbai %.4f R@100 %.4f' % (a['nDCG@10'], a['Recall@100'], b['nDCG@10'], b['Recall@100']), flush=True)
PY
done
