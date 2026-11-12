#!/usr/bin/env bash
# P0 #2, honest version: retrain the SFT cross-encoder while DUMPING its
# seed-determined held-out split, then
#   (a) rerank exactly those held-out queries -> fit the fusion weight out-of-fold
#   (b) rerank dev with the same checkpoint -> the number we publish
#
# Why retrain: the 2026-11-09 e1/e2 checkpoints never recorded which 45 train
# queries they held out, and replaying the split from the current code does not
# reproduce the logged docs_loaded fingerprint, so any "recovered" held-out set
# is a guess. 21 min of training buys a provable split.
#
#   bash scripts/phase11_sft_oof.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/d/anaconda/python.exe}; export PYTHONUTF8=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TAG=${TAG:-v2m3_pool8_e2b}
SPLIT=${SPLIT:-results/ce_val45/val_split_e2b.json}
LOG=logs/phase11_sft_oof_${TAG}.log
exec >>"$LOG" 2>&1
echo "=== phase11 start tag=$TAG $(date +%H:%M:%S)"
DOMS="bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace"

# 1) dump the split once, then train on exactly that split
$PY scripts/train_ce.py --domains $DOMS --groups-prefix results/ce_train_groups \
  --run-tag mxbai --tag dump_split --val-frac 0.12 --val-queries 4 --val-depth 100 \
  --seed 17 --dump-val-qids "$SPLIT" || { echo "FAIL dump"; exit 1; }
$PY scripts/train_ce.py --domains $DOMS --groups-prefix results/ce_train_groups \
  --model models/bge-reranker-v2-m3 \
  --run-tag mxbai --tag "$TAG" --epochs 2 --lr 1.5e-5 --warmup-frac 0.08 \
  --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
  --val-frac 0.12 --val-queries 4 --val-depth 100 --seed 17 \
  --val-qids "$SPLIT" || { echo "FAIL train"; exit 1; }
echo "--- split dumped $(date +%H:%M:%S)"

# 2) held-out examples for the out-of-fold fusion fit
$PY - "$SPLIT" <<'PYEOF'
import json, os, sys
split = json.load(open(sys.argv[1], encoding="utf-8"))
outdir = "results/ce_val45_e1b"
os.makedirs(outdir, exist_ok=True)
for d, qids in split["eval"].items():
    keep = set(qids)
    rows = []
    with open(f"data/track1_tempo/{d}/examples_train.jsonl", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            if r["id"] in keep:
                rows.append(json.dumps(r, ensure_ascii=False))
    with open(f"{outdir}/examples_{d}.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
print("examples written", {d: len(v) for d, v in split["eval"].items()})
PYEOF

# 3) rerank the held-out queries -> out-of-fold fusion weight
for d in $DOMS; do
  ex=results/ce_val45_e1b/examples_${d}.jsonl
  sub=runs/subset_e1b_${d}_train.trec
  out=runs/rerank_e1b_${d}_oof.trec
  [ -s "$out" ] && { echo "skip $d"; continue; }
  $PY scripts/make_subset_run.py --run runs/dense_mxbai_${d}_train.trec \
    --examples "$ex" --out "$sub" >/dev/null || { echo "FAIL subset $d"; continue; }
  $PY scripts/run_chunk_rerank.py --domain "$d" --split oof \
    --examples "$ex" --run "$sub" --k 100 \
    --model "results/ce_ft/$TAG/best" --q-mode title --q-cap 256 --win 250 --stride 192 \
    --pool max --batch 32 --out "$out" \
    --meta results/rerank_e1b_${d}_oof.meta.json || echo "FAIL oof $d"
done
$PY scripts/fit_fusion_weight.py --split train \
  --dense-pattern "runs/subset_e1b_{domain}_train.trec" \
  --ce-pattern "runs/rerank_e1b_{domain}_oof.trec" \
  --select-metric domain_macro --out results/fusion_fit_oof_e2b.json
echo "--- oof fit done $(date +%H:%M:%S)"

# 4) dev rerank with the same checkpoint (the published number)
CKPT="results/ce_ft/$TAG/best" TAG=sft_e2b bash scripts/phase9_dev_rerank.sh
echo "=== phase11 done $(date +%H:%M:%S)"
