# Phase 11 validation — out-of-fold fusion weight for the SFT reranker

Status: complete (protocol frozen before the dev number was read, 2026-11-09; both arms finished 23:25).

## Question

Phase 9 reported domain-macro 0.3336 for the SFT CE arm fused at the zero-shot weight (RRF 1:1).
That number inherited a weight chosen for the zero-shot arm, and the 1,211 training queries
share 45 queries with the fusion-fit split, so the arm had never been scored with a weight that
was chosen without those queries.

Phase 11 asks the narrow, falsifiable version:

> With the fusion weight fitted **out of fold** on the 45 held-out training queries, does the
> SFT CE arm still beat the zero-shot CE arm on dev by at least +0.005 domain-macro nDCG@10?

## Protocol (frozen)

1. `train_ce.py --dump-val-qids results/ce_val45/val_split_e1b.json` picks the same 45 train
   queries (seed 17, 4 per domain, depth 100) and writes them out.
2. Train `v2m3_pool8_e1b` on the remaining 1,166 train queries:
   `bge-reranker-v2-m3`, epochs 1, lr 1e-5, warmup 0.08, freeze 8 layers, max-len 320,
   windows max 3, pool-chunk 32, seed 17. The base model must be `bge-reranker-v2-m3`
   (567.8 M parameters); `scripts/check_ce_model.py --tag v2m3_pool8_e1b --expect v2-m3`
   guards against the 278 M `bge-reranker-base` default (see §Incident below).
3. Build the 45-query held-out rerank inputs from the frozen dense mxbai top-1000 runs and
   rerank them with the SFT checkpoint (chunked CE, q-mode title, q-cap 256, win 250, stride 192).
4. Fit the fusion weight on those 45 queries over the grid rrf 1:1, rrf 2:1, z 1:1, z 1:2 and
   keep the best (`scripts/fit_fusion_weight.py`, results/fusion_fit_oof_e1b.json).
5. Apply that weight, unchanged, to the full dev rerank of the same checkpoint (1,021 dev
   queries, 13 domains) and compare with the zero-shot arm at its own best weights.

Pass rule (fixed before step 5 was run): adopt the OOF weight whatever it is; keep the SFT
claim in the paper iff `SFT_OOF − zero_shot_best ≥ +0.005` domain-macro, otherwise the SFT
arm is reported as "no measurable gain" and the headline falls back to the zero-shot arms.

## Expected failure modes

- The 45 held-out queries are small (3–4 per domain), so the weight may be noisy; that is why
  the rule fixes the weight first and only then looks at dev.
- 13 domains × 45 queries means several domains contribute 1 query; domain-macro is therefore
  high variance. Report both domain-macro and topic-macro, and both pooled and domain-clustered
  bootstrap intervals.

## Incident (2026-11-09, recorded for reproducibility)

The first Phase 11 attempt silently ran with the wrong base model: `train_ce.py` defaults
`--model` to `models/bge-reranker-base` (278.0 M parameters) and `phase11_sft_oof.sh` did not
pass `--model`. The resulting CE arm collapsed to 0.1957 domain-macro (zero-shot CE: 0.2680)
and every derived number (dev 0.2799, OOF fit 0.3182) measured that 278 M model, not the
method. Those artefacts were moved to `results/_wrong_base_20261109/`, the script now passes
`--model models/bge-reranker-v2-m3`, and `scripts/check_ce_model.py` fails a run whose
`train_log.jsonl` shows a different model or a parameter count below 500 M.

A second complication: orphaned jobs from the previous turn's queue (variant training
`v2m3_pool8_e2s23`/`e2s29`, `ops/watch_then_*.py`, `phase12_*`/`phase13_*` scripts) kept
taking the 8 GB GPU and killing the Phase 11 training step with a hard abort. The queue's
drivers and watchers were disabled (moved to `ops/_disabled_20261109/`,
`scripts/_train_sft_variant.sh.disabled`) before this run.

Third: the training must run as a managed job. `nohup ... &` children of a tool call were
reaped mid-epoch on this host (two attempts died at step ≈200 with no traceback); the same
command started with `run_in_background` completed.

## Results

All dev numbers are 13 domains / 519 topics, official domain-macro nDCG@10, and
were re-scored from the current run files after the staleness fix described
below.

### Out-of-fold fit (45 held-out train queries)

| form | weights | held-out domain-macro |
|---|---|---|
| rrf | 1:4 | **0.3023** (selected, e1b) / **0.3265** (e2b) |
| rrf | 1:8 | 0.3006 / 0.3239 |
| z | 1:2 | 0.2997 (e1b) |
| rrf | 1:1 | 0.2921 / 0.3234 |
| z | 1:1 | 0.2814 (e1b) |

Both fits prefer a dense-heavy RRF weight (1:4). This is a statement about the
weight, not about reranker quality: no dev run was read during either fit.

### Dev, by arm

| arm | CE alone | RRF 1:4 (OOF) | RRF 1:1 | z 1:1 |
|---|---:|---:|---:|---:|
| zero-shot `bge-reranker-v2-m3` | 0.2680 | 0.2981 | 0.3166 | 0.3249 |
| SFT 1 epoch (`v2m3_pool8_e1b`) | 0.2728 | 0.3074 | 0.3172 | 0.3340 |
| SFT 2 epochs (`v2m3_pool8_e2b`) | 0.2753 | 0.3187 | 0.3431 | **0.3583** |
| SFT 2 epochs, seed 23 (`v2m3_s23_e2`) | 0.2793 | – | – | 0.3575 |
| SFT 2 epochs, seed 29 (`v2m3_s29_e2`) | 0.2778 | – | – | 0.3528 |
| frozen 2-epoch arm (`sub_sft_e2bc`) | 0.2823 | – | – | 0.3581 |
| dense mxbai only | – | – | – | 0.2943 |

Matched-weight, domain-clustered bootstraps against the zero-shot arm at the
same fusion form:

| comparison | Δ domain-macro | 95% CI | better/worse/tied |
|---|---:|---|---|
| e2b z 1:1 vs zs z 1:1 | **+0.0334** | [+0.0191,+0.0477] | 186/79/254 |
| seed 23 z 1:1 | +0.0325 | [+0.0184,+0.0470] | 186/83/250 |
| seed 29 z 1:1 | +0.0279 | [+0.0160,+0.0405] | 182/83/254 |
| e1b z 1:1 | +0.0091 | [+0.0043,+0.0137] | 92/52/375 |
| e2b RRF 1:1 | +0.0265 | [+0.0052,+0.0473] | 182/94/243 |
| e2b RRF 1:4 (OOF) | +0.0205 | [+0.0010,+0.0404] | 181/122/216 |
| e1b RRF 1:4 (OOF) | +0.0094 | [+0.0008,+0.0191] | 112/79/328 |
| e1b RRF 1:1 | +0.0006 | [−0.0064,+0.0082] | 104/63/352 |

Robust views for the reported arm (z 1:1): 12 domains (iota dropped) 0.3181 vs
0.2832 (+0.0349); the 8 domains with at least 19 queries 0.3267 vs 0.2985
(+0.0282). The gain is not carried by the three-query iota domain.

### Pre-registered decision

Literal bar: SFT at its out-of-fold weight (e2b 0.3187) minus the zero-shot best
configuration (z 1:1 0.3249) = **−0.0062**, so "SFT beats the whole zero-shot
system at a never-dev-fitted weight" **fails**. Arm-level bar: SFT vs zero-shot
at the same fusion form is **+0.0205 … +0.0334** with domain CIs excluding zero,
at the OOF weight, at RRF 1:1 and at z 1:1.

The headline is therefore:

- **0.3581 reproduces**: a fresh checkpoint of the same 2-epoch recipe scores
  0.3583 and two other seeds 0.3575 / 0.3528 at z 1:1 (spread 0.0055 across
  seeds, an order of magnitude below the +0.033 gain). 0.3581 stays the system
  number; 0.3431 is the same arm under the V1-fixed RRF 1:1 form.
- The SFT reranker is reliably better than the zero-shot one at a fixed form;
  the fusion *weight* chosen on the 45 held-out train queries (RRF 1:4) is
  worse than the inherited z 1:1 form, so weight refitting does not improve the
  system, it only confirms the gain is not a weight artefact.

### Incident 2: stale score files produced a false retraction

The first version of this document reported e1b/e2b z 1:1 as 0.3117 / 0.3196
and concluded that the frozen 0.3581 did not reproduce. That conclusion was
wrong, and the cause was mundane: `results/score_sub_sft_e{1,2}b_*_dev.json`
had been written from *intermediate* fused TREC files, the fused files were
later regenerated at the end of the rerank stage, and the score files were never
recomputed. Re-scoring the current files gives 0.3340 / 0.3583.

Guard added: `scripts/check_score_freshness.py` walks `results/score_*.json`,
compares each file's mtime with the `run_file` recorded inside it, and lists the
stale ones (`--fix` re-scores them). An audit of the current tree reports zero
stale files after the re-score.

## Files

- `results/fusion_fit_oof_e1b.json`, `results/fusion_fit_oof_e2b.json` — OOF fits
- `runs/rerank_sft_e{1,2}b_<domain>_dev.trec`, `runs/rerank_sft_s{23,29}_<domain>_dev.trec`
- `runs/fused_sft_e{1,2}b_<domain>_dev.trec` (z 1:1), `runs/fused_sft_e{1,2}b_rrf_1{1,4}_<domain>_dev.trec`
- `results/aggregate_sub_sft_e{1,2}b_dev.json`, `results/aggregate_sub_sft_s{23,29}_dev.json`,
  `results/aggregate_sft_e{1,2}b_rrf_1{1,4}_dev.json`, `results/aggregate_ce_sft_*_dev.json`
- `results/boot_sub_sft_e2b_vs_sub_zs11.json`, `results/boot_sub_sft_e1b_vs_sub_zs11.json`,
  `results/boot_sft_s{23,29}_vs_zs.json`, `results/boot_sft_e2b_vs_zs_rrf1{1,4}.json`
