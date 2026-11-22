cd RETECO
rm -f runs/dense_qwen3emb06_*.trec /tmp/reteco/emb/qwen3emb06_*.npy
setsid nohup env CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/dense_qwen3.py \
  --model /homeC/wujiaheng/models-qwen3-emb06 \
  --domains bitcoin quant law --batch 48 --tag qwen3emb06 \
  > logs/qwen3emb_fix.log 2>&1 < /dev/null &
sleep 25
tail -3 logs/qwen3emb_fix.log
pgrep -f dense_qwen3.py | head -2
