cd /homeC/wujiaheng/RETECO; pgrep -af watch_emb.sh | head -2; cat logs/emb_watch.log 2>/dev/null | tail -3; pgrep -af dense_qwen3.py | head -1; date +%H:%M
