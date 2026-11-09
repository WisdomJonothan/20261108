#!/usr/bin/env bash
# Wait for the SFT run, apply the pre-registered gate, and (only if it passes)
# start the dev re-ranking automatically so no GPU time is wasted.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
TAG=${TAG:-v2m3_pool8_e1}
LOG=results/ce_ft/${TAG}/train_log.jsonl
GATE=results/ce_ft/${TAG}/gate.json
echo "=== watch $LOG $(date +%H:%M:%S)"
while ! grep -q '"event": "done"' "$LOG" 2>/dev/null; do sleep 60; done
$PY - "$LOG" "$GATE" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
z = [r for r in rows if r.get("event") == "epoch0_zeroshot"][0]
e = [r for r in rows if r.get("event") == "epoch"][-1]
d_auc = e["val_auc"] - z["val_auc"]
d_ndcg = e["val_ndcg10"] - z["val_ndcg10"]
gate = {"zero_shot_auc": z["val_auc"], "tuned_auc": e["val_auc"], "delta_auc": d_auc,
        "zero_shot_ndcg10": z["val_ndcg10"], "tuned_ndcg10": e["val_ndcg10"],
        "delta_ndcg10": d_ndcg, "n_val_queries": e["n_val_queries"],
        "threshold": 0.01, "passed": bool(d_auc >= 0.01)}
json.dump(gate, open(sys.argv[2], "w"), indent=1)
print(json.dumps(gate))
PY
PASS=$($PY -c "import json;print(1 if json.load(open('$GATE'))['passed'] else 0)")
echo "=== gate passed=$PASS $(date +%H:%M:%S)"
if [ "$PASS" = "1" ]; then
  CKPT=results/ce_ft/${TAG}/best TAG=sftv1 bash scripts/phase9_dev_rerank.sh
  echo "=== dev rerank finished $(date +%H:%M:%S)"
else
  echo "=== gate failed: keep V1.1, SFT reported as an ablation"
fi
