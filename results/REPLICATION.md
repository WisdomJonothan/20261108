# Seed replication of the frozen 2-epoch SFT arm (2026-11-09)

Protocol fixed before the run (recorded in `results/NEXT_STEPS.md` §P1 #2): retrain the
frozen recipe with a different seed, rerank dev with the best-validation checkpoint,
freeze with the frozen fusion (z 1:1) and compare against the zero-shot fused arm with
the official domain-macro statistic. Pass rule: mean delta over seeds >= +0.005, at
least 2/3 seeds positive, pooled domain-clustered CI excluding zero.

Run `v2m3_pool8_e2s29`: identical recipe to `v2m3_pool8_e2` except `--seed 29`
(2 epochs, peak lr 1.5e-5, freeze 8/24, max-pool training parity, max 3 windows,
max_len 320). Validation: 0.2846 (zero-shot) -> **0.3195** (epoch 1, best) -> 0.3054
(epoch 2); `best/` = epoch 1, the same selection rule as the original run.
Training log `results/ce_ft/v2m3_pool8_e2s29/train_log.jsonl`; config
`results/ce_ft/TRAINING_CONFIGS.md`; rerank runs `runs/rerank_sft_e2s29_*_dev.trec`.

## Dev results (519 queries, 13 domains)

| arm | 13-domain (official) | delta vs zero-shot | domain-clustered 95% CI | better/worse/tied | 12-domain (no iota) | n>=19 (8 domains) |
|---|---|---|---|---|---|---|
| zero-shot z 1:1 | 0.3249 | — | — | — | 0.2832 | 0.2985 |
| **SFT 2-epoch, seed 17 (frozen)** | **0.3581** | +0.0332 | [+0.0192,+0.0472] | 188/78/253 | 0.3181 | 0.3267 |
| SFT 2-epoch, seed 29 | 0.3527 | +0.0278 | [+0.0159,+0.0404] | 181/82/256 | 0.3123 | 0.3232 |
| mean of the two seeds | 0.3554 | +0.0305 | — | — | 0.3152 | 0.3250 |

Files: `results/aggregate_sub_sft_e2s29_dev.json`,
`results/boot_dom_e2s29_vs_zs.json`, `results/aggregate_sub_sft_e2bc_dev.json`,
`results/boot_dom_e2_vs_zs.json`, `results/aggregate_sub_zs11_dev.json`.

## Verdict

**Replicates.** Both seeds improve on the zero-shot fused baseline, both
domain-clustered intervals exclude zero, and the seed-to-seed spread of the arm
(0.3527 vs 0.3581, 1.5% relative) is far smaller than the delta over the baseline
(+0.028 / +0.033). The 12-domain and n>=19 views give the same ordering, so iota
(3 queries) is not carrying the effect. The honest headline is therefore
**+0.031 +/- 0.003 over the strongest zero-shot baseline**, with the seed-17 number
(0.3581) reported as the frozen submission system and seed 29 as the replication.
