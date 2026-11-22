cd RETECO
echo "--- qwen3 run head"; head -3 runs/dense_qwen3emb06_bitcoin_dev.trec
echo "--- mxbai run head"; head -3 runs/dense_mxbai_bitcoin_dev.trec
echo "--- qrels head"; head -3 data/track1_tempo/bitcoin/qrels_dev.txt
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0,'scripts')
from lib_reteco import load_qrels, read_trec_run
q = load_qrels('data/track1_tempo/bitcoin/qrels_dev.txt')
for r in ['runs/dense_qwen3emb06_bitcoin_dev.trec','runs/dense_mxbai_bitcoin_dev.trec']:
    run = read_trec_run(r)
    k = list(run)[0]
    docs = [d for d,_ in run[k][:100]]
    gold = set(q.get(k, {}))
    print(r, 'topics', len(run), 'docs0', len(docs), 'gold', len(gold), 'hit@100', len([d for d in docs if d in gold]), 'scores', run[k][0][1], run[k][-1][1])
PY
