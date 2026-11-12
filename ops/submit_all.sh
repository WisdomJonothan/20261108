#!/usr/bin/env bash
# Build a complete, validated submission package for every test domain that
# has a query file, and record provenance for each artefact.
#
#   bash ops/submit_all.sh <test-queries-dir> <checkpoint> [outdir]
#   DOMAINS="iota law" bash ops/submit_all.sh ...   # subset
#
# For each domain: make_submission (dense -> CE -> z 1:1 -> de-tie) ->
# official format_checker -> strict local sanity -> md5 + config into MANIFEST.
set -u
cd "$(dirname "$0")/.." || exit 1
# the official checker opens files with the locale codec; on a Chinese
# Windows that is GBK and it dies on UTF-8 corpora
export PYTHONUTF8=1
PY=${PY:-/d/anaconda/python.exe}
TEST_DIR=${1:?usage: submit_all.sh <test-queries-dir> <checkpoint> [outdir]}
CKPT=${2:?missing checkpoint}
OUT=${3:-results/submission_pkg}
# WORK_ROOT lets a test run reuse an existing cache directory; the
# default keeps every package self-contained.
WORK_ROOT=${WORK_ROOT:-$OUT}
mkdir -p "$OUT" "$WORK_ROOT"
fail=0
built=""

for f in "$TEST_DIR"/examples_test_*.jsonl; do
  [ -e "$f" ] || { echo "no examples_test_*.jsonl under $TEST_DIR"; exit 1; }
  d=$(basename "$f" | sed 's/^examples_test_//; s/\.jsonl$//')
  if [ -n "${DOMAINS:-}" ]; then
    case " $DOMAINS " in *" $d "*) ;; *) continue ;; esac
  fi
  echo "=== $d"
  run="$OUT/submission_$d.trec"
  work="$WORK_ROOT/work_$d"
  if ! "$PY" scripts/make_submission.py --domain "$d" --queries "$f" \
        --index-data "data/track1_tempo/$d" --model-ce "$CKPT" \
        --fusion z --wa 1 --wb 1 --out "$run" --work "$work" \
        > "$OUT/build_$d.log" 2>&1; then
    echo "!!! $d: make_submission failed (see $OUT/build_$d.log)"; fail=1; continue
  fi
  if ! "$PY" scripts/official/format_checker.py "$run" \
        --corpus "data/track1_tempo/$d/documents.jsonl" > "$OUT/check_$d.txt" 2>&1; then
    echo "!!! $d: official format_checker FAILED"; tail -3 "$OUT/check_$d.txt"; fail=1; continue
  fi
  if ! "$PY" scripts/check_submission_strict.py "$run" "$f" > "$OUT/strict_$d.txt" 2>&1; then
    echo "!!! $d: strict sanity FAILED"; tail -3 "$OUT/strict_$d.txt"; fail=1; continue
  fi
  echo "    $(grep -c . "$run") lines, checker + strict sanity OK"
  built="$built $d"
done

if [ -z "$built" ]; then echo "nothing built"; exit 1; fi

"$PY" - "$OUT" $built <<'PY'
import hashlib, json, os, sys
out, domains = sys.argv[1], sys.argv[2:]
man = {"domains": {}, "domains_built": domains}
for d in domains:
    run = os.path.join(out, "submission_%s.trec" % d)
    cfg = run + ".config.json"
    h = hashlib.md5(open(run, "rb").read()).hexdigest()
    entry = {"run": run, "md5": h, "lines": sum(1 for _ in open(run, encoding="utf-8"))}
    if os.path.isfile(cfg):
        c = json.load(open(cfg, encoding="utf-8"))
        entry["config"] = {k: c.get(k) for k in
                           ("queries", "q_sha1", "fusion", "wa", "wb", "rrf_wa", "rrf_wb",
                            "ce_q_mode", "ce_q_cap", "ce_win", "ce_pool", "topk",
                            "dense_fp", "ce_fp", "model_ce")
                           if k in c}
    man["domains"][d] = entry
man["fail"] = False
json.dump(man, open(os.path.join(out, "MANIFEST.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({d: (e["lines"], e["md5"][:8]) for d, e in man["domains"].items()}, indent=1))
PY
echo "=== manifest: $OUT/MANIFEST.json"
exit $fail
