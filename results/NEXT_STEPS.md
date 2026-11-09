# Next steps — decision record (2026-11-09)

Status: dev-only, frozen best = `runs/sub_sft11_*` (domain-macro 0.3336). Every
number below is the **official domain-macro** metric (per-query nDCG@10, mean
within domain, mean over 13 domains; missing topics = 0).

## 1. Where the system stands

| system | domain-macro | topic-macro |
|---|---|---|
| official BM25 | 0.1147 | — |
| dense mxbai top-100 | 0.2943 | 0.2891 |
| + zero-shot CE, RRF 1:1 (V1) | 0.3166 | 0.3199 |
| + zero-shot CE, z-sum 1:1 | 0.3249 | 0.3260 |
| + SFT CE, z-sum 1:1 | **0.3336** | 0.3339 |
| SFT vs zero-shot at fixed 1:1 | +0.0087 [+0.0037, +0.0142] (domain-clustered) | |
| SFT total vs V1 | +0.0170 [+0.0052, +0.0287] | |

Headroom is in scoring, not recall *in the aggregate*: oracle@10 inside the
frozen top-100 is 0.7446 and the submitted system captures **8.7%** of the gap
(41% of queries improve, 19% worsen). The CE's within-query AUC (0.80) is not
better than the dense retriever's, which is why supervised training — not more
candidate recall — is the only lever that moved the number. Locally, though,
recall is decisive: 27.7% of dev queries score 0 and **36% of those have no gold
doc in the top-100 at all** (`results/DEV_ERROR_ANALYSIS.md`).

## 2. GPT review P0 — status

| # | item | status |
|---|---|---|
| 1 | unify the metric on domain-macro | DONE — `score_run.py`, `aggregate_scores.py`, `bootstrap_compare.py --stat domain`, `fit_fusion_weight.py --select-metric`; refit picks z 1:1 |
| 2 | refit the SFT fusion weight out-of-fold | CLOSED BY DECISION (C′) — the e1/e2 held-out split is not recoverable (replaying it from the current code does not reproduce the logged `docs_loaded` counts, all 13 domains, seeds 0–60), so a 45-query refit would be a 45-query guess (SE ≈ 0.02 ≈ 2× the effect). Instead: keep the inherited 1:1 (fitted on train with the **control** arm, which can only understate SFT), disclose it, and prove weight-robustness with a both-arms dev sweep — `scripts/weight_sweep.py`, `results/WEIGHT_SWEEP.md`. Retrain-with-dumped-split remains available as `scripts/phase11_sft_oof.sh` if a reviewer demands a provable fold |
| 3 | `score_run.py` missing-topic zero-fill | DONE (on GitHub at 6be4e80) |
| 4 | `validate_submission.py --detie` re-validates its output | DONE — old `-1e-9*rank` underflowed on the padding sentinel and left 900 ties/topic; replaced with an ordinal fix |
| 5 | `test_mode_dryrun.sh` overwrite guard | DONE — refuses without FORCE=1 |
| 6 | reproduction material | DONE — `requirements.txt`, `results/MODEL_MANIFEST.json` (sizes+sha256), `results/ce_ft/TRAINING_CONFIGS.md`, `scripts/reproduce_all.sh`, both `train_log.jsonl` |

## 3. What to do next, in order

**P1 — finish the two-week story (no new modelling).**
1. *Out-of-fold fusion weight for SFT.* The 2026-11-09 e1/e2 checkpoints did not
   record which train queries they held out, and the split cannot be replayed
   (documented in `results/ce_val45/val_split_seed17_true.json`). `scripts/phase11_sft_oof.sh`
   therefore retrains the same config with the split dumped by the run itself
   (`--dump-val-qids`), reranks those held-out queries, fits the weight on them,
   and only then reranks dev. Budget: 21 min train + 6 min OOF rerank + ~1 h dev
   rerank, GPU-serial. Expected effect ±0.005 (a weight change cannot move AUC;
   the 8.7%-capture law applies).
   Pass rule: adopt the OOF weight whatever it is; the SFT-vs-zero-shot delta
   must stay >= +0.005 at that same weight. Cost ~0.3 GPU-h.
2. *Seed replication of the only significant gain.* Two more single-epoch runs
   (seeds 23, 29; otherwise identical to `v2m3_pool8_e1`) + dev rerank each.
   Keep SFT in the title iff mean delta >= +0.005, >= 2/3 seeds positive, pooled
   domain-clustered CI excludes 0. Cost ~2.7 GPU-h train + ~2 h rerank.
3. *Synthetic test split replay.* Copy dev, rewrite ids, run dense -> CE ->
   fusion -> official `format_checker` end to end. Prevents a submission-day
   surprise; produces the exact command for January. Cost 0 GPU-h (~1 h).
4. *Report the robust views as primary alongside the headline:* 12-domain
   (SFT 0.2918 vs V1 0.2782) and n>=19 (0.3043 vs 0.2913) — the headline leans
   on iota, which has 3 dev queries.

**P2 — evidence for the paper (CPU-only, highest value per hour).**
5. *Annotate 100 year-mismatched gold documents* (blinded, 30 double-coded) to
   turn "mismatch is not irrelevance" from an inference into a table: what are
   those documents (retrospective, retrospective + new development, wrong year
   in text)? This is the paper's spine and it costs 0 GPU-hours.
6. *Rerun the temporal-ceiling analysis on the submitted mxbai top-100* (the
   file currently analyses bge-base candidates) and fix its 6/8 vs 7/8 miscount.

**Kill list (do not spend time on):** per-domain fusion weights (fitted on
train: negative), the 1.5B reranker (OOM risk on 8 GB, expected +~0.002), more
hand-built temporal features (ten families already null), any document-ID
feature (leaks the retrieval pipeline).

## 4. Paper in one line

"Temporal consistency is not the binding constraint in RETECO Track 1a: text-
extractable time evidence is too sparse and time mismatch is not irrelevance, so
a temporal-agnostic two-view retriever-reranker with train-fitted fusion is the
strongest 8 GB-viable system, and the remaining headroom is scorer
discrimination, not temporal modelling."

Sections: intro/thesis -> benchmark anatomy (observability) -> system ladder ->
fusion protocol -> supervised reranking -> temporal null + ceiling -> why
nothing else moves -> limitations/reproducibility. Claims to cut or requalify:
"ten pre-registered negatives" (only the gate has a frozen protocol), "temporal
signals do not work" (only no gain under our protocol), "fusion is better" (CI
crosses zero), "SFT validated" (one seed).

## 5. Update 2026-11-09 ~14:40 (afternoon)

- **A second SFT checkpoint (2 epochs, lr 1.5e-5, `v2m3_pool8_e2`) is better than
  the frozen one.** Val was inconclusive (0.3029 vs 0.3008), but the first four
  completed dev domains give e2 − e1 = +0.0058 bitcoin, +0.0211 cardano,
  +0.0355 economics, +0.0222 genealogy (105 queries, 34 up / 14 down). Full
  13-domain rerank is running (job bash-1236, tag `sft_e2bc`); if the aggregate
  holds, e2 replaces e1 as the frozen submission arm — freeze with
  `TAG=sft_e2bc bash scripts/freeze_sft_arm.sh` (fusion → de-tie+validate →
  score → aggregate in one command).
- **Depth analysis** (`results/DEPTH_ANALYSIS.md`): dense Recall@300 = 0.8183
  (0.6995 at 100) and the pool's oracle nDCG@10 rises 0.7446 → 0.8513, i.e. 22.5 %
  of gold docs sit at ranks 100–1000. A 62-query probe (`scripts/run_depth_probe.sh`)
  decides whether the CE's 8.7 % capture survives at depth; the decision rule
  (+0.01 on the probe, CI excluding 0) is fixed in the doc before the run.
- **Paper draft** now at `docs/PAPER_DRAFT.md` (447 lines, every number carries
  its source file in an HTML comment); a verification pass against the artifacts
  is in flight.
- Submission path is test-phase ready: `scripts/make_submission.py` passes
  `--examples` to both stages, keys the CE cache by checkpoint, de-ties inside
  the writer; runbook in `docs/SUBMISSION.md`.
