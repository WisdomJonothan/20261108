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
can be regenerated deterministically with `--seed 17` (recovered and stored in
`results/ce_val45/val_qids.json`).

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
PYTHONUTF8=1 /d/anaconda/python.exe scripts/train_ce.py \
  --domains bitcoin cardano economics genealogy history hsm iota law monero politics quant travel workplace \
  --groups-prefix results/ce_train_groups --run-tag mxbai --val-frac 0.12 --val-queries 4 --val-depth 100 \
  --freeze-layers 8 --window-mode max --max-windows 3 --max-len 320 --pool-chunk 32 \
  --epochs 1 --lr 1e-5 --seed 17 --out-dir results/ce_ft/v2m3_pool8_e1
```
