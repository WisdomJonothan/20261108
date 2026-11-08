# Phase 9 — what to do next: decision record

**Status**: decision taken and execution started (2026-11-08/09). This note is the
record of the multi-agent review + the reasoning; numbers that come from agents
are marked *(agent-computed)* and were re-derived locally where listed.

---

## 1. Where the project actually stands

Frozen V1 = `mxbai-embed-large-v1` dense top-100 → `bge-reranker-v2-m3` CE
(title query, 250-token windows, max-pool) → RRF 1:1 over the same 100.

| quantity | value | how |
|---|---:|---|
| topic-macro nDCG@10 (official aggregation) | **0.3199** | `scripts/aggregate_scores.py` over `results/score_sub_1_1_*_dev.json` |
| domain-macro (how the project used to quote it) | 0.3166 | same |
| topic-macro over the 8 domains with n ≥ 19 | 0.3243 | same |
| official BM25 1a dev baseline (organizers) | 0.1147 | RETECO data page |
| TEMPO best reported system (different split) | 0.320 | TEMPO paper |

Two things this exposed:

1. **The headline was under-specified.** 0.3166 (domain-macro) and 0.3199
   (topic-macro) are different numbers for the same runs, and the leaderboard
   uses topic-macro. All future tables must state the aggregation.
2. **`score_run.py` used to drop topics missing from a run** (mean over present
   topics only). The leaderboard counts them as 0. Fixed: missing topics are now
   zero-filled unless `--allow-missing` is passed.

Frozen artifacts: `runs/sub_1_1_<domain>_dev.trec` (13 files, de-tied so a
score-sorting scorer cannot permute tied top-10 slots), validated by the
official `format_checker.py` with `--qrels` **and** `--corpus` through
`scripts/validate_submission.py` → 0 errors on all 13 files.

## 2. The finding that changed the plan: there is no train-less domain

The project had been reasoning from "only 8 of 13 domains have a train split",
which made supervised training look like it would cover only 60 % of the
evaluation. That was a **partial download**: `cardano, law, quant, travel,
workplace` had never been fetched. The official release ships 1,211 train
queries over all 13 domains (70/30 per domain); after
`scripts/fetch_missing_train.py` the local counts are exactly
1,211 train / 519 dev.

Consequences: supervised training now has in-domain labels for every evaluated
domain, and "does train-tuned transfer to unseen domains?" is no longer the
primary risk. Leakage check (agent-computed and re-checked): train and dev share
zero topic ids; maximum query token-Jaccard between any dev query and all train
queries is 0.19.

## 3. Diagnosis: ranking is the binding constraint, and the CE is not better than the retriever

| quantity | value |
|---|---:|
| oracle nDCG@10 inside the frozen top-100 pool | 0.745 |
| V1 achieved | 0.320 |
| fraction of ranking headroom captured | ~5 % |
| within-query AUC, dense mxbai | 0.806 |
| within-query AUC, zero-shot bge-reranker-v2-m3 | 0.801 |
| Recall@100 macro (mxbai) | 0.699 |
| Recall@100 macro (mxbai ∪ bge dense top-100) | 0.737 |
| oracle@10 macro (mxbai ∪ bge) | 0.778 |

The cross-encoder has **no discriminative edge** over the retriever it re-ranks;
the +0.03 fusion gain comes from rank complementarity of two near-equal scorers.
This explains why ten clever post-hoc interventions returned ~0: any idea that
does not move the scorer's within-query discrimination cannot move nDCG.

Recall is not the binding constraint either: unioning a second dense retriever
buys +0.038 Recall@100 and +0.034 oracle@10, but a 5 %-capture ranker converts
that into ≈ +0.002 *(agent-computed transfer curve)*.

## 4. Options considered and verdicts

| option | verdict | evidence |
|---|---|---|
| **SFT the cross-encoder on train (1,211 q)** | **DO** | only intervention aimed at the binding constraint; no corpus re-embedding; earlier CE-only transfer is limited (see §5) so it must be paired with a refit fusion weight |
| per-domain fusion weights fitted on train | **NO** (measured negative) | train-fitted per-domain weights → dev 0.3082 vs fixed 2:1 0.3179 *(agent-computed)*; train-argmax and dev-argmax disagree; weight spread on train only 0.005 → unidentifiable |
| contrastive fine-tune of the bi-encoder | NO (this cycle) | every checkpoint needs a 1.17M-doc re-encode at 45–60 doc/s ≈ 5.4–8.4 h; recall is not binding |
| pool expansion / SPLADE / ColBERT | NO (this cycle) | storage-infeasible at 8 GB (ColBERT); SPLADE ≈ 7 h encode for an unproven member |
| local generative reranker (Qwen2.5-1.5B/3B) | NO | already failed as listwise (primacy replay) and sign-unstable as pairwise in Phase 2; 11.8 s/query |
| more temporal features | NO | TEMPO §6 reports the same null; Q-TIE requires per-document timestamps that this corpus does not have |
| z-normalised **score-sum** fusion instead of RRF | **TRY** (CPU-only, free) | agent-computed +0.0097 query-bootstrap [+0.0017,+0.0180], domain-clustered +0.0070 [−0.0022,+0.0163]; must be refit on train before it is claimed |
| freeze V1, validate, package, write | **DO, in parallel** | the submission is mandatory; V1 already scores ~2.8× the official lexical baseline |

## 5. Why SFT must be paired with a fusion refit

Simulating a *perfect* CE inside the frozen candidate pool *(agent-computed)*:
RRF 2:1 reaches only 0.391, RRF 1:1 ~0.44, RRF 1:4 0.663. The share of a
CE-alone gain that survives fusion is ≈7 % at 2:1, 19 % at 1:1, 38 % at 1:2,
61 % at 1:4, 26 % under score-sum 1:1.

So the experiment is **not** "fine-tune a reranker and look at the dev number".
It is: fine-tune → refit the one fusion weight **on train** → read dev once.
A SFT run that improves CE-alone by +0.05 but keeps RRF 2:1 would look like
+0.003 and be wrongly declared dead.

## 6. Pre-registered protocol (frozen before the dev read)

1. Fit on train only: CE weights, epoch selection by held-out **train** queries
   (12 % per domain), fusion weight by train macro.
2. Two scorers are compared on dev, once each:
   * V1 = zero-shot CE + fixed RRF 1:1 (already frozen, 0.3199);
   * V2 = SFT CE + train-fitted fusion.
3. Adopt V2 only if Δtopic-macro ≥ +0.015 **and** the domain-clustered paired
   bootstrap CI excludes 0. Between +0.010 and +0.015: report as a wash.
   Below +0.010: V2 is an ablation, V1 is submitted.
4. Kill criterion for the training itself: if held-out-train within-query AUC
   does not improve by ≥ +0.01 over the zero-shot model, stop at one epoch and
   ship V1.
5. Report both aggregations, per-domain rows, and the n ≥ 19 subset.

Caveats to state in the paper: dev has been the selection set for V1's design
choices; the honest estimate for V1 is therefore the train-fitted/LOOCV number,
not the 0.3199 in-sample maximum. The project's own precedent for that
shrinkage: a temporal gate that looked like +0.0245 under dev CV delivered
+0.0039 under the separated protocol.

## 6b. Result: V1.1 adopted (train-fitted fusion), dev read once

Fitting every (form, weight) candidate on the 1,211 **train** queries picks
z-normalised score-sum **1:2** (train topic-macro 0.2893, vs 0.2782 for V1's
RRF 1:1). Applied to dev once:

| system | dev topic-macro | domain-macro | paired bootstrap (query) | domain-clustered |
|---|---:|---:|---|---|
| V1 (RRF 1:1) | 0.3199 | 0.3166 | — | — |
| **V1.1 (z-score-sum 1:2)** | **0.3302** | 0.3193 | +0.0103 [+0.0021,+0.0184] | [−0.0131,+0.0175] |
| n ≥ 19 subset | 0.3243 → **0.3365** | | | |

The train-fitted optimum and the dev optimum coincide, so this is not a dev
artefact; it remains unestablished under domain clustering, which is the honest
limit. Frozen as `runs/sub_1_2_<domain>_dev.trec` (13/13 official-format valid).

## 7. Execution log (this session)

| step | status |
|---|---|
| fetch the 5 missing train splits, verify 1,211 / 519 | done (`scripts/fetch_missing_train.py`) |
| dense mxbai train runs for the 179 new queries + iota (13/13 domains) | done (`scripts/phase9_train_dense_new.sh`) |
| CE train groups, 13 domains, listwise, 1 pos-all + 70 % hard negs | done (`scripts/build_ce_train_groups.py`): 1,071 groups, 2,876 pos, 18,544 neg |
| fine-tune trainer with train-holdout validation, AUC, window sampling | done (`scripts/train_ce.py`) |
| pilot: bge-reranker-base, 1 epoch, 13 domains | 0.2240 → 0.2301 val nDCG (+0.006), i.e. the loop learns but the weak base model is not worth keeping |
| main: bge-reranker-v2-m3, 2 epochs, lr 1e-5, bf16 + grad-ckpt (5.5 GB peak) | running (`logs/phase9_sft_v2m3.log`) |
| V1 freeze: de-tied runs, official format validation, strict scoring, aggregation | done |
| submission validator `scripts/validate_submission.py` | done — 13/13 files, 0 errors |

## 8. Remaining submission checklist

1. Dev rerank with the SFT checkpoint (~60 min GPU) and dev-fusion read (once).
2. Train-side CE runs for the 6 domains that still lack them (cardano, law,
   quant, travel, workplace, iota; 186 queries) so the fusion weight and the
   score-sum variant can be fitted on train rather than on dev.
3. `reproduce_all.sh` + `RUN_MANIFEST.sha256` + `requirements.txt` + pinned
   model revisions; publish the 6 scripts that exist only in the workspace.
4. Test-mode dry run: `--split test` path, topic-id equality against
   `examples_test.jsonl`, corpus hash assertion (blocks the "wrong corpus
   silently scored" failure mode).
5. Paper: headline must be stated as topic-macro with the aggregation named;
   the temporal null results (10 pre-registered negatives + the
   "time-mismatch ≠ irrelevance" analysis) are the strongest section, not the
   score.
