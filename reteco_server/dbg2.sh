cd RETECO
echo "lines:"; wc -l runs/dense_qwen3emb06_bitcoin_dev.trec
echo "md5 of first 3 lines:"; head -3 runs/dense_qwen3emb06_bitcoin_dev.trec | od -c | head -6
.venv/bin/python - <<'PY'
import json
p='data/track1_tempo/bitcoin/documents.jsonl'
n=0
with open(p,encoding='utf-8') as fh:
    for line in fh:
        d=json.loads(line); n+=1
        if n<=2: print({k:(str(v)[:80]) for k,v in d.items()})
print('docs', n)
PY
