#!/usr/bin/env bash
# One-command reproduction of the frozen RETECO Track 1a dev system.
#
#   bash scripts/reproduce_all.sh            # full 13-domain dev run (~70 min on the 4060)
#   DOMAINS=iota bash scripts/reproduce_all.sh   # one domain, ~2 min
#
# Stages: dense (mxbai top-100) -> CE rerank (v2-m3, zero-shot or CKPT) ->
#         z-score fusion 1:1 -> official validation (de-tied) -> nDCG@10.
#
# ENV:
#   CKPT     fine-tuned checkpoint dir (default: the zero-shot model)
#   TAG      output tag (default: repro_zs / repro_sft)
#   DOMAINS  space-separated domain list (default: all 13)
#   PY       python interpreter (default /d/anaconda/python.exe)
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}
export PYTHONUTF8=1
DOMAINS=${DOMAINS:-"bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"}
MODEL=${CKPT:-models/bge-reranker-v2-m3}
TAG=${TAG:-$( [ "$MODEL" = "models/bge-reranker-v2-m3" ] && echo repro_zs || echo repro_sft )}
OUT=runs/repro_${TAG}
mkdir -p "$OUT" results/repro_"$TAG"

echo "=== reproduce tag=$TAG model=$MODEL $(date +%H:%M:%S)"
for d in $DOMAINS; do
  echo "--- $d dense"
  $PY scripts/run_dense_generic.py --index-data "data/track1_tempo/$d" --split dev \
    --model models/mxbai-embed-large-v1 --tag mxbai --pooling mean \
    --query-prefix "Represent this sentence for searching relevant passages: " \
    --work emb --out "$OUT/dense_${d}.trec" --meta "$OUT/dense_${d}.json"

  echo "--- $d cross-encoder"
  $PY scripts/run_chunk_rerank.py --domain "$d" --split dev \
    --run "$OUT/dense_${d}.trec" --k 100 --model "$MODEL" \
    --q-mode title --q-cap 256 --win 250 --stride 192 --pool max --batch 32 \
    --out "$OUT/ce_${d}.trec" --meta "$OUT/ce_${d}.meta.json"

  echo "--- $d fusion 1:1 (train-selected under the official domain-macro metric)"
  $PY scripts/fuse_scores.py --run-a "$OUT/dense_${d}.trec" --run-b "$OUT/ce_${d}.trec" \
    --wa 1 --wb 1 --norm z --out "$OUT/fused_${d}.trec" --json "$OUT/fuse_${d}.json"

  echo "--- $d validate (de-tied)"
  $PY scripts/validate_submission.py --runs "$OUT/fused_${d}.trec" --split dev \
    --detie "$OUT/sub_${d}.trec" --json "$OUT/validation_${d}.json"

  echo "--- $d score"
  $PY scripts/score_run.py --run "$OUT/sub_${d}.trec" --data "data/track1_tempo/$d" \
    --split dev --json "results/repro_${TAG}/score_${d}.json" >/dev/null
done
$PY scripts/aggregate_scores.py --pattern "results/repro_${TAG}/score_*.json" --tag "$TAG" \
  --json "results/repro_${TAG}/aggregate.json"
echo "=== done $(date +%H:%M:%S)  -> results/repro_${TAG}/aggregate.json"
