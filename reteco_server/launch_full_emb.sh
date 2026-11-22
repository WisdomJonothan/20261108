cd /homeC/wujiaheng/RETECO
setsid nohup env CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/dense_qwen3.py \
  --model /homeC/wujiaheng/models-qwen3-emb06 --batch 48 --tag qwen3emb06 \
  --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
  > logs/qwen3emb_full.log 2>&1 < /dev/null &
echo launched
