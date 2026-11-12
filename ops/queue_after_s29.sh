#!/usr/bin/env bash
# Runs after the seed-replication script finishes, GPU-serial with it.
#   1) Phase 13 test-split replay on GPU (law quant iota), the January dry run.
#   2) Dev rerank with the epoch-1 checkpoint of the Phase 11 refit
#      (results/ce_ft/v2m3_pool8_e2b/epoch1) -- tests the hypothesis that the
#      0.3196 outlier came from that run selecting epoch 2.
# Started detached (Start-Process), started 2026-11-10 ~02:20.
cd /d/RETECO || exit 1
export PYTHONUTF8=1
LOG=logs/queue_after_s29.log
exec >>"$LOG" 2>&1
echo "=== queue start $(date +%H:%M:%S) pid=$$"

for i in $(seq 1 300); do
  if grep -q "=== seed repl done" logs/phase11_seed_repl.log 2>/dev/null; then
    echo "seed repl done detected $(date +%H:%M:%S)"
    break
  fi
  sleep 60
done
grep -q "=== seed repl done" logs/phase11_seed_repl.log \
  || echo "WARN: seed repl still not done after 300 min; proceeding anyway"
sleep 30

echo "--- phase13 replay $(date +%H:%M:%S)"
bash scripts/phase13_test_replay.sh law quant iota || echo "FAIL phase13"

echo "--- e2b epoch1 dev rerank $(date +%H:%M:%S)"
CKPT=results/ce_ft/v2m3_pool8_e2b/epoch1 TAG=sft_e2bep1 bash scripts/phase9_dev_rerank.sh \
  || echo "FAIL rerank"
TAG=sft_e2bep1 bash scripts/freeze_sft_arm.sh || echo "FAIL freeze"

echo "=== queue done $(date +%H:%M:%S)"
