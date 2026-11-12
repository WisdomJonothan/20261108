# Reproducing the frozen system from nothing

This is the path the paper's 0.3581 row comes from. Every stage writes files with
a fingerprint sidecar, so a re-run either reproduces or refuses — it never
silently reuses a mismatched cache.

Hardware used: one RTX 4060 Laptop (8 GB), 15.8 GB RAM, Windows 11. Wall-clock
numbers below are from that machine; the GPU steps are the only slow ones.

```bash
export PYTHONUTF8=1          # official tools open files with the locale codec
PY=/d/anaconda/python.exe    # the interpreter that has torch 2.7.0+cu126 + transformers
```

## 0. Data

Official release, unpacked to `data/track1_tempo/<domain>/` with
`documents.jsonl`, `examples_{train,dev}.jsonl`, `qrels_{train,dev}.txt`,
`qrels_steps_{train,dev}.txt` for each of the 13 domains.
`results/corpus_manifest.json` records 1,167,159 deduplicated documents.
`scripts/fetch_missing_train.py` exists because an early download was missing the
train shards of cardano/law/quant/travel/workplace.

## 1. Dense first stage (GPU, ~2-3 h for the full corpus)

```bash
for d in <13 domains>; do
  $PY scripts/run_dense_generic.py --index-data data/track1_tempo/$d --split dev \
      --model models/mxbai-embed-large-v1 --tag mxbai --pooling mean \
      --query-prefix "Represent this sentence for searching relevant passages: " \
      --topk 100 --work emb --out runs/dense_mxbai_${d}_dev.trec \
      --meta results/dense_mxbai_${d}_dev.json
done
```

Writes `runs/dense_mxbai_<domain>_dev.trec` plus a `doc_emb_*.npy.fp.json`
sidecar binding model, pooling, max length, dtype and document count. A cache
without a sidecar is adopted once (provenance `adopted-legacy`) and refused
afterwards if the inputs change. Expected dev: **0.2943** domain-macro.

## 2. Supervised cross-encoder (GPU, ~55 min for 2 epochs on an RTX 4060 laptop)

```bash
$PY scripts/build_ce_train_groups.py --domains <13 domains> --run-tag mxbai --k 100
# -> results/ce_train_groups_<domain>.jsonl (dense top-100 candidates + labels)
$PY scripts/train_ce.py --domains <13> \
    --model models/bge-reranker-v2-m3 \
    --loss possum --sampler uniform --q-mode title --q-cap 256 \
    --win 250 --stride 192 --window-mode max --max-windows 3 --max-len 320 \
    --pool-chunk 32 --freeze-layers 8 --epochs 2 --lr 1.5e-5 --warmup-frac 0.08 \
    --seed 17 --val-frac 0.12 --val-queries 4 --val-metric mean \
    --val-qids results/val_qids_e2.json \
    --tag v2m3_pool8_e2
```

`best/` is chosen on the cross-encoder's own ordering over held-out training
queries: `--val-frac 0.12` holds out 12% of each domain's groups and
`--val-queries 4` scores at most four of them per domain, which is the 45-query
split used here. `--val-qids results/val_qids_e2.json` pins exactly which queries
(the fraction-based split depends on a shuffle). The frozen artifact selected
epoch 1 on the plain mean (0.3029 against 0.3012); that run predates domain-macro
logging. The current code writes both numbers every epoch and
`--val-metric domain_macro` selects on the official aggregation instead; on every
later arm the two criteria pick the same epoch, so no reported number depends on
which one is chosen.

## 3. Re-rank dev + fuse + score (GPU, ~35 min; CPU, seconds)

```bash
bash ops/final_eval.sh sft_e2bc results/ce_ft/v2m3_pool8_e2/best title 256 1 1
```

which is, per domain: `run_chunk_rerank_qvar.py` -> `fuse_scores.py --norm z
--wa 1 --wb 1` -> `validate_submission.py --detie` -> `score_run.py` ->
`aggregate_scores.py` -> `bootstrap_compare.py` against zero-shot, against the
frozen arm and `dev_multievidence.py`.

Expected: frozen **0.3581**, zero-shot CE z 1:1 **0.3249**, dense **0.2943**,
paired gain **+0.0332** [+0.0192, +0.0472].

## 4. Submission

See `docs/SUBMISSION.md` (runbook + `ops/submit_all.sh` + manifest). The
synthetic test replay is reproducible byte for byte:
`WORK_ROOT=results/test_replay` rebuilds the frozen iota file with
md5 `a210d92a9f06317cbb2a17adf22387a2`.

## What is *not* reproducible, and why

* The 12% train-internal split used before `--val-qids` existed depended on a
  shuffle seeded by the code path; `results/val_qids_e2.json` pins the split used
  from 2026-11-10 onward.
* The Phase 11 out-of-fold weights were fit on 45 queries (SE ~0.02); they are
  reported as a sensitivity check, not as the system.
* Seed ensembling was tried and rejected (`results/ENSEMBLE.md`).
