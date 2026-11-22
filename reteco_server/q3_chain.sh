cd /homeC/wujiaheng/RETECO
DENSE=${DENSE:-dense_qwen3emb06}
TAG=${TAG:-q3ce}
for d in bitcoin quant law; do
  if [ ! -s runs/${TAG}_${d}_dev_ce.trec ]; then
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/run_chunk_rerank_qvar.py \
      --domain $d --split dev --run runs/${DENSE}_${d}_dev.trec --k 100 \
      --q-mode title --q-cap-words 0 --pair-max-len 510 --q-tokens 256 \
      --win 250 --stride 192 --pool max \
      --model results/ce_ft/v2m3_pool8_e2/best \
      --out runs/${TAG}_${d}_dev_ce.trec --meta results/reteco/${TAG}_${d}_dev.meta.json --batch 16 \
      >> logs/q3_ce.log 2>&1
  fi
  .venv/bin/python scripts/fuse_scores.py --dense runs/${DENSE}_${d}_dev.trec \
    --ce runs/${TAG}_${d}_dev_ce.trec --out runs/${TAG}_${d}_dev.trec --norm z --wa 1 --wb 1 >> logs/q3_ce.log 2>&1
  .venv/bin/python scripts/score_run.py --run runs/${TAG}_${d}_dev.trec --data data/track1_tempo/$d --split dev \
    --json results/${TAG}/score_${d}.json > /dev/null 2>&1
  echo "$d $(.venv/bin/python -c "import json;d=json.load(open('results/${TAG}/score_${d}.json'));print('fused %.4f'%d['nDCG@10'])")"
done
echo Q3_DONE
