#!/usr/bin/env bash
# Test-mode dry run: exercise the full pipeline on a split named `test`.
#
# The real test split is released during the evaluation window (10-31 Jan 2027)
# as `examples_test.jsonl` in the same domain directories, with no qrels. This
# script proves the code path works before that: it fabricates `examples_test`
# from `examples_dev` for two small domains, runs dense -> CE -> fusion ->
# validation with `--split test`, and then removes the fabricated files.
#
# It checks the failure modes that are invisible in dev scoring:
#   * a topic-id that does not match the examples file (scores 0 officially)
#   * a run that misses topics (officially counted as 0, not dropped)
#   * a document id that is not in the domain corpus
#   * missing qrels (the official checker must fall back to structure-only)
#
# Usage:  bash scripts/test_mode_dryrun.sh [domains...]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}
export PYTHONUTF8=1
DOMAINS=${*:-"iota law"}
OUT=tmp/test_dryrun
LOG=logs/test_mode_dryrun.log
mkdir -p "$OUT" logs
exec >>"$LOG" 2>&1

echo "=== test-mode dry run $(date +%H:%M:%S)  domains: $DOMAINS"
rc=0
for d in $DOMAINS; do
  ex="data/track1_tempo/$d/examples_test.jsonl"
  if [ -e "$ex" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "REFUSE $d: $ex already exists (real test split?). Set FORCE=1 to override."
    rc=1; continue
  fi
  echo "--- $d: fabricating examples_test from examples_dev"
  cp "data/track1_tempo/$d/examples_dev.jsonl" "$ex" || { echo "FAIL cp $d"; rc=1; continue; }

  $PY scripts/run_dense_generic.py --index-data "data/track1_tempo/$d" --split test \
    --model models/mxbai-embed-large-v1 --tag mxbai --pooling mean \
    --query-prefix "Represent this sentence for searching relevant passages: " \
    --work emb --out "$OUT/dense_mxbai_${d}_test.trec" \
    --meta "$OUT/dense_mxbai_${d}_test.json" || { echo "FAIL dense $d"; rc=1; }

  $PY scripts/run_chunk_rerank.py --domain "$d" --split test \
    --run "$OUT/dense_mxbai_${d}_test.trec" --k 100 \
    --model models/bge-reranker-v2-m3 --q-mode title --q-cap 256 --win 250 --stride 192 \
    --pool max --batch 32 --out "$OUT/rerank_mxbai_${d}_test.trec" \
    --meta "$OUT/rerank_mxbai_${d}_test.meta.json" || { echo "FAIL ce $d"; rc=1; }

  $PY scripts/fuse_scores.py --run-a "$OUT/dense_mxbai_${d}_test.trec" \
    --run-b "$OUT/rerank_mxbai_${d}_test.trec" --wa 1 --wb 2 --norm z \
    --out "$OUT/sub_1_2_${d}_test.trec" --json "$OUT/fuse_${d}_test.json" \
    || { echo "FAIL fuse $d"; rc=1; }

  # qrels_test.txt does not exist: the checker must still validate structure,
  # topic ids against examples_test and doc ids against the corpus.
  $PY scripts/validate_submission.py --runs "$OUT/sub_1_2_${d}_test.trec" \
    --split test --json "$OUT/validation_${d}_test.json" || { echo "FAIL validate $d"; rc=1; }

  rm -f "$ex"   # the guard above guarantees this is our fabricated copy
done
echo "=== dry run done rc=$rc $(date +%H:%M:%S)"
exit $rc
