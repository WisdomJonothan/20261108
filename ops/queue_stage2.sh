#!/usr/bin/env bash
# Stage 2 of the ladder (user memo m00374, steps 1-3).
#
# Waits for the four-arm objective comparison, picks the arm with the best
# train-internal domain-macro score, then measures the four query
# representations on that arm's checkpoint only (title / body / full / ctxt /
# summ at a shared word budget). Nothing here reads dev qrels: the dev
# re-ranking happens later, once, for the winner.
set -u
cd /d/RETECO || exit 1
LOG=logs/stage2.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

PY=/d/anaconda/python.exe
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DOMS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"

# ------------------------------------------------------------ 1. wait
waited=0
until grep -q "=== loss experiment done" logs/loss_exp.log 2>/dev/null; do
  sleep 60
  waited=$((waited+1))
  if [ $waited -ge 480 ]; then
    log "timeout: four-arm comparison did not finish within 8h"
    exit 1
  fi
done
log "four-arm comparison finished (waited ${waited}m)"

# ------------------------------------------------- 2. pick the winner
$PY - 2>&1 <<'PY' | tee -a "$LOG"
import json, os

rows = []
for tag in sorted(os.listdir("results/ce_ft")):
    if not tag.startswith("lx_"):
        continue
    p = os.path.join("results", "ce_ft", tag, "train_log.jsonl")
    if not os.path.isfile(p):
        continue
    e0, best = None, None
    for ln in open(p, encoding="utf-8"):
        r = json.loads(ln)
        if r.get("event") == "epoch0_zeroshot":
            e0 = r
        elif r.get("event") == "best":
            best = r
    if best is None:
        rows.append({"tag": tag, "score": None, "note": "no best row"})
        continue
    score = best.get("score", best.get("val_ndcg10"))
    rows.append({"tag": tag, "score": round(score, 6),
                 "epoch": best.get("epoch"),
                 "domain_macro": round(best.get("val_ndcg10_domain_macro", 0.0), 6),
                 "mean": round(best.get("val_ndcg10", 0.0), 6),
                 "epoch0_mean": round(e0.get("val_ndcg10", 0.0), 6) if e0 else None})
rows.sort(key=lambda r: (r["score"] is not None, r["score"] or 0.0), reverse=True)
winner = rows[0]["tag"] if rows else None
# Tie band: the 45-query internal split resolves only ~0.03 (see
# docs/RESEARCH_PLAN_NEXT.md 5b), so an arm must beat the incumbent loss by
# at least TIE_BAND on val to replace it; ties go to the simpler arm.
TIE_BAND = 0.010
BASE_TAG = "lx_possum"
_by = {r["tag"]: r for r in rows if r.get("score") is not None}
forced = None
if winner and winner != BASE_TAG and BASE_TAG in _by:
    _margin = _by[winner]["score"] - _by[BASE_TAG]["score"]
    if _margin < TIE_BAND:
        forced = {"from": winner, "to": BASE_TAG, "margin": round(_margin, 6)}
        winner = BASE_TAG
with open("results/LOSS_WINNER.json", "w", encoding="utf-8") as f:
    json.dump({"metric": "train-internal val domain-macro nDCG@10", "rows": rows,
               "tie_band": TIE_BAND, "forced": forced,
               "winner": winner}, f, indent=1)
print(json.dumps({"winner": winner, "rows": rows}, indent=1))
PY

WINNER=$($PY -c "import json;print(json.load(open('results/LOSS_WINNER.json'))['winner'] or '')")
if [ -z "$WINNER" ]; then
  log "no arm produced a best checkpoint; stopping"
  exit 1
fi
log "winner: $WINNER"

# ------------------------------------- 3. query representation comparison
$PY scripts/train_ce.py \
  --domains $DOMS --groups-prefix results/ce_train_groups \
  --model models/bge-reranker-v2-m3 \
  --init-from "results/ce_ft/$WINNER/best" \
  --val-qids results/val_qids_e2.json \
  --eval-qvariants title,body,full,ctxt,summ --q-cap-words 64 \
  --q-mode title --q-cap 256 --win 250 --stride 192 \
  --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
  --dtype fp16 --eval-win-limit 2 --val-metric domain_macro \
  --tag "qv_$WINNER" > "logs/qvar_$WINNER.log" 2>&1 || log "FAIL qvariants"

$PY - "logs/qvar_$WINNER.log" 2>&1 <<'PY' | tee -a "$LOG"
import json, sys

path = sys.argv[1]
res = None
for ln in open(path, encoding="utf-8"):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    if r.get("event") == "qvariants":
        res = r
print(json.dumps(res, indent=1) if res else "no qvariants event found")
if res:
    rows = [{"mode": m, **v} for m, v in res["results"].items()]
    rows.sort(key=lambda r: r["domain_macro"], reverse=True)
    json.dump({"cap_words": res.get("cap_words"), "rows": rows,
               "winner": rows[0]["mode"] if rows else None},
              open("results/QVARIANT_WINNER.json", "w", encoding="utf-8"), indent=1)
PY

# ------------------------------------------------------------- 4. summary
$PY scripts/summarize_arms.py --runs lx_possum lx_allpos lx_ndcg lx_sqrt \
  --baseline '' --out results/LOSS_ARMS.md >> "$LOG" 2>&1 || log "FAIL summarize"

log "=== stage2 done"
