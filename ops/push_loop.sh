#!/usr/bin/env bash
# Retry the push until origin/main == HEAD (network on this host is flaky).
# Detached from the job tree, so it is not killed by the ~10-minute job cap.
set -u
cd /d/RETECO/github_repo || exit 1
for i in $(seq 1 120); do
  out=$(git push origin main 2>&1)
  if [ $? -eq 0 ]; then
    echo "$(date +%H:%M:%S) push ok on attempt $i" >> logs/push_loop.log
    exit 0
  fi
  echo "$(date +%H:%M:%S) attempt $i failed: $(printf '%s' "$out" | tail -1)" >> logs/push_loop.log
  sleep 90
done
echo "$(date +%H:%M:%S) giving up" >> logs/push_loop.log
exit 1
