#!/usr/bin/env bash
# Phase 13 — synthetic test-split replay through the *submission* code path.
#
# The January risk is not the model: it is the one script that must turn hidden
# test queries into a valid TREC run.  This script fabricates an
# examples_test.jsonl from dev queries for a few small domains, runs
# `make_submission.py` on it exactly as it will run in January, and validates
# the output with the official checker (corpus mode; no qrels exist for test).
#
# Usage:  bash scripts/phase13_test_replay.sh [DOMAIN ...]
#         (default: law workplace quant iota)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY=${PY:-D:/anaconda/python.exe}
DOMAINS=("$@")
[ ${#DOMAINS[@]} -eq 0 ] && DOMAINS=(law workplace quant iota)
CKPT=${CKPT:-results/ce_ft/v2m3_pool8_e2/best}
OUT=${OUT:-results/test_replay}
mkdir -p "$OUT"

fail=0
for d in "${DOMAINS[@]}"; do
  src="data/track1_tempo/$d/examples_dev.jsonl"
  ex="$OUT/examples_test_$d.jsonl"
  run="$OUT/submission_$d.trec"
  echo "=== $d: building synthetic test queries -> $ex"
  "$PY" - "$src" "$ex" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
with open(dst, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({"id": r["id"], "query": r["query"],
                            "domain": r.get("domain", "")}, ensure_ascii=False) + "\n")
print(f"  {len(rows)} queries")
PY

  echo "=== $d: make_submission (dense -> CE -> z 1:1 -> de-tie)"
  "$PY" scripts/make_submission.py --domain "$d" \
    --queries "$ex" --index-data "data/track1_tempo/$d" \
    --model-ce "$CKPT" --fusion z --wa 1 --wb 1 \
    --out "$run" --work "$OUT/work_$d"

  echo "=== $d: official format_checker (corpus mode)"
  if ! "$PY" scripts/official/format_checker.py "$run" \
        --corpus "data/track1_tempo/$d/documents.jsonl"; then
    echo "!!! $d FAILED the official checker"; fail=1; continue
  fi

  echo "=== $d: local sanity (10 docs/topic, contiguous ranks, strict scores)"
  "$PY" - "$run" "$ex" <<'PY' || fail=1
import json, sys
run, ex = sys.argv[1], sys.argv[2]
qids = [json.loads(l)["id"] for l in open(ex, encoding="utf-8")]
rows = {}
for line in open(run, encoding="utf-8"):
    p = line.split()
    rows.setdefault(p[0], []).append((int(p[3]), float(p[4])))
missing = [q for q in qids if q not in rows]
bad = []
for q in qids:
    r = rows.get(q, [])
    if len(r) != 10: bad.append((q, f"{len(r)} rows"))
    if [x[0] for x in r] != list(range(1, 11)): bad.append((q, "ranks"))
    if any(r[i][1] <= r[i+1][1] for i in range(len(r)-1)): bad.append((q, "scores"))
if missing or bad:
    print("  SANITY FAIL", {"missing": len(missing), "bad": bad[:5]}); sys.exit(1)
print(f"  OK: {len(qids)} topics x 10 docs, ranks 1-10, strictly decreasing scores")
PY
done

echo
if [ "$fail" -eq 0 ]; then echo "PHASE 13: PASS"; else echo "PHASE 13: FAILURES ABOVE"; exit 1; fi
