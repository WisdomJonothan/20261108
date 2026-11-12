#!/bin/bash
# Deferred tail job: after the main ladder finishes, give the two best
# loss/sampling arms their own dev numbers.
#
# The four screening arms are compared on 45 train-internal queries, which is
# +/-0.01 noisy -- too coarse to decide anything. Once the critical path
# (stage2 -> step arms -> stage3 -> final dev eval) is over and the GPU is free,
# score the top-2 arms by domain-macro on dev so the paper can report the loss
# ablation on the official split too.
set -u
cd /d/RETECO || exit 1
PY=/d/anaconda/python.exe
LOG=logs/lossdev_tail.log
: > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# wait for the stage3 marker (or give up after 10 h)
for i in $(seq 1 600); do
  if grep -q "=== stage3 done" logs/stage3.log 2>/dev/null; then
    log "stage3 finished; starting loss-arm dev evals"
    break
  fi
  if [ "$i" -eq 600 ]; then
    log "stage3 never finished; giving up"
    exit 0
  fi
  sleep 60
done

sleep 30
$PY - <<'PY' >> "$LOG" 2>&1
import glob, io, json, os
rows = []
for p in glob.glob("results/ce_ft/lx_*/train_log.jsonl"):
    tag = os.path.basename(os.path.dirname(p))
    best = None
    for ln in io.open(p, encoding="utf-8"):
        e = json.loads(ln)
        if e.get("event") == "best":
            best = e
    if best:
        rows.append({"tag": tag, "epoch": best.get("epoch"),
                     "score": best.get("score"),
                     "mean": best.get("val_ndcg10")})
rows.sort(key=lambda r: -(r["score"] or 0))
print(json.dumps({"event": "loss_arms_ranked", "rows": rows}, indent=1))
picked = [r["tag"] for r in rows[:2]]
with io.open("results/LOSSDEV_PICKS.json", "w", encoding="utf-8") as f:
    json.dump({"picked": picked, "rows": rows, "rule": "top-2 by val domain_macro"},
              f, indent=1)
print("picked:", picked)
PY

PICKS=$($PY -c "import json;print(' '.join(json.load(open('results/LOSSDEV_PICKS.json'))['picked']))" 2>/dev/null)
log "picks: $PICKS"

for tag in $PICKS; do
  if [ -f "results/aggregate_sub_${tag}_dev.json" ]; then
    log "skip $tag (already scored)"
    continue
  fi
  log "=== final_eval $tag"
  bash ops/final_eval.sh "$tag" "results/ce_ft/${tag}/best" title 256 1 1 >> "$LOG" 2>&1 \
    || log "FAIL final_eval $tag"
done

log "=== lossdev tail done"
