cd RETECO
for r in dense_mxbai_bitcoin_dev dense_qwen3emb06_bitcoin_dev; do
  .venv/bin/python scripts/score_run.py --run runs/$r.trec --data data/track1_tempo/bitcoin --split dev --json /tmp/$r.json >/dev/null 2>&1
  echo "$r $(.venv/bin/python -c "import json;d=json.load(open('/tmp/$r.json'));print(round(d['nDCG@10'],4), round(d['Recall@100'],4), round(d['oracle_nDCG@10_cand100'],4))")"
done
date +%H:%M
