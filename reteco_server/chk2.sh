cd RETECO; ls -l logs/emb_watch.log 2>/dev/null; pgrep -af watch_emb.sh | head -2; cat logs/emb_watch.log 2>/dev/null | tail -3; date +%H:%M
