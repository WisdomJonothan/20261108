#!/bin/bash
# Wait for the condition-window arm, then run the two-window arm on the same GPU.
cd "$HOME/RETECO" || exit 1
for i in $(seq 1 180); do
  [ -f results/replacement_v2/exp_cond_win_dev.json ] && break
  sleep 30
done
if [ ! -f results/replacement_v2/exp_cond_win_dev.json ]; then
  echo "cond arm never finished" >> logs/two_win.log
  exit 1
fi
setsid env CUDA_VISIBLE_DEVICES=3 nohup .venv/bin/python -u scripts/localized_judge.py \
  --root . --select two --batch-size 16 \
  --out results/replacement_v2/exp_two_win_dev.json > logs/two_win.log 2>&1 < /dev/null &
echo "two-window arm launched"
