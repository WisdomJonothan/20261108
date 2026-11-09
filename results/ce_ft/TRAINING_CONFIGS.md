# Supervised cross-encoder runs (bge-reranker-v2-m3)

Base model: `BAAI/bge-reranker-v2-m3` (568M, 24 XLM-R layers), local `models/bge-reranker-v2-m3`.
Training data: `results/ce_train_groups_<domain>.jsonl` from
`scripts/build_ce_train_groups.py` over 1,211 train queries → 1,071 usable
groups, 2,876 positives, 18,544 negatives (70 % hard negatives mined from the
dense top-100).  Only in-candidate positives are used; dev gold documents that
reappear in a train candidate pool are treated as unlabelled negatives, so the
supervision never touches dev labels.

Held-out selection: `--val-frac 0.12` of *groups* are removed from training,
then up to `--val-queries 4` per domain of those queries are scored at depth 100
(45 queries: bitcoin 4, cardano 4, economics 4, genealogy 4, history 4, hsm 4,
iota 1, law 3, monero 4, politics 4, quant 2, travel 4, workplace 3).  The ids
cannot be regenerated from the current code (the `docs_loaded` fingerprint does
not reproduce); see `results/NEXT_STEPS.md` §5. The corrected rerun dumps them with
`--dump-val-qids`.

Common settings: listwise softmax over the candidate group, bf16 autocast,
gradient checkpointing, AdamW, 8 % linear warmup, weight decay 0.01,
`--freeze-layers 8` (202.6M trainable of 567.8M), `--window-mode max`
(= deployed inference pooling), `--max-windows 3`, `--max-len 320`,
`--pool-chunk 32`, seed 17, batch = 1 group.

| run dir | epochs | lr (peak) | val nDCG@10 (epoch 0 → best) | val AUC |
|---|---|---|---|---|
| `v2m3_pool8_e1` | 1 | 1e-5 | 0.2870 → 0.3008 | 0.7950 → 0.7990 |
| `v2m3_pool8_e2` | 2 | 1.5e-5 | 0.2750 → 0.3029 (epoch 1) / 0.3012 (epoch 2) | 0.7843 → 0.7909 |

`train_log.jsonl` in each run dir records every epoch, the LR schedule and the
per-domain validation nDCG.  Checkpoints (`best/`, `epoch1/`, `epoch2/`) are
~2.2 GB each and are not committed; the frozen dev reranks produced from them
are.

Reproduce one run:

```bash
PYTHONUTF8=1 /d/anaconda/python.exe scripts/train_ce.py --model models/bge-reranker-v2-m3 \
  --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
  --groups-prefix results/ce_train_groups --run-tag mxbai --val-frac 0.12 --val-queries 4 --val-depth 100 \
  --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
  --epochs 1 --lr 1e-5 --seed 17 --tag v2m3_pool8_e1 --outdir results/ce_ft
```

## Run `v2m3_pool8_e2s29` (seed replication, 2026-11-09)

Same recipe as `v2m3_pool8_e2` (2 epochs, peak lr 1.5e-5, warmup 0.08, freeze 8/24 =
202.6M trainable, `--window-mode max --max-windows 3 --max-len 320 --pool-chunk 32`,
bf16, grad-checkpointing, `--val-frac 0.12 --val-queries 4 --val-depth 100 --pool-size` not used)
with **`--seed 29`**. Trained detached (`ops/launch_train29.ps1`) so it survived the
host's ~10-minute job kill.

Validation trajectory (held-out train queries, same size as the original run):

| epoch | val nDCG@10 | val AUC |
|---|---|---|
| 0 (zero-shot) | 0.2846 | 0.8189 |
| 1 | **0.3195** (best) | 0.7924 |
| 2 | 0.3054 | 0.7890 |

`best/` = epoch-1 checkpoint (`{"event": "best", "epoch": 1, "val_ndcg10": 0.3195434}`),
the same checkpoint-selection rule that picked epoch 1 for the original run
(0.3029 vs 0.3012). Log: `results/ce_ft/v2m3_pool8_e2s29/train_log.jsonl`.
