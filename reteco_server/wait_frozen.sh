#!/usr/bin/env bash
# Wait for the metric-aligned CE dev run to finish, then reproduce the frozen checkpoint
# through the identical chain so the two are comparable.
set -u
cd "$HOME/RETECO" || exit 1
for i in $(seq 1 240); do
  if grep -q "NDCG_DEV_DONE ndcgdev" logs/ndcg_chain.out 2>/dev/null; then
    bash ndcg_dev_chain.sh 0 frozenrepro frozen > logs/frozen_chain.out 2>&1
    exit 0
  fi
  sleep 30
done
echo "TIMEOUT waiting for ndcgdev"
