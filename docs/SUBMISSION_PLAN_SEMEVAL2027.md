# SemEval 2027 RETECO system paper — submission plan

Status vocabulary used throughout (do not blur these):

* **draft** — exists locally, not reviewed by anyone outside this project.
* **submitted** — sent to the SemEval 2027 system-description track.
* **accepted** — notification received.
* **published** — camera-ready online (ACL Anthology), after the workshop.

As of 2026-11-13 the project is at **draft**. Nothing has been submitted. The
RETECO 2027 evaluation window (test queries) opens 2027-01-10 and closes
2027-01-31; system-description papers are due 2027-02 (exact dates pending the
organisers' announcement).

## 1. What the venue expects

A SemEval system description is reviewed on: system design, data splits,
experimental setup, ablations, and error analysis. Leaderboard rank is not the
only criterion (see GUIR at SemEval-2026, a system paper that reports a
mid-table system with a careful analysis). The paper must be reproducible from
released artefacts, and its claims must match what the released code produces.

Our current 4-page draft already has: system design (§3), splits and metrics
(§2), the full baseline ladder with paired bootstraps (§4), five ablations
(§5.2-§5.8), and a negative-results narrative. What it does not yet have is
listed in §3 below.

## 2. Page budget (ACL two-column, 4 pages + references for a system paper)

| part | now | target |
|---|---|---|
| abstract | 300 w | 200 w |
| 1 introduction + contributions | 210 w | 250 w |
| 2 task, metric, data | 330 w | 300 w |
| 3 system | 900 w | 800 w |
| 4 results + baseline ladder | 700 w | 550 w |
| 5 analysis (5.1-5.8) | 2100 w | 1500 w |
| 6 reproducibility and efficiency | 250 w | 250 w |
| 7 limitations | 350 w | 200 w |
| 8 error analysis (new) | 0 | 300 w |
| 9 conclusion (new) | 0 | 100 w |

The analysis section is over budget: §5.1, §5.4 and §5.6 can each lose a third
of their prose without losing a number. §5.8 (CE-vs-fusion) is the newest and
should keep its table.

## 3. Gaps before "submittable"

1. **Weights.** Six checkpoints, 1.08 GB each, are not released and have no
   download address (`results/CHECKPOINTS.md`). A reviewer must be able to
   reproduce 0.3581. Decision needed: publish `v2m3_pool8_e2` (the frozen
   system) and `cx_qv` first, on ModelScope (reachable from this machine) or
   HuggingFace (needs a mirror), with md5 in the model card.
2. **Test predictions.** The official test queries do not exist yet. Until the
   evaluation window opens, the paper can only report dev numbers; §4 and the
   abstract must say "dev" explicitly, and the submission must add the official
   test score once released.
3. **Error analysis section.** `results/DEV_ERROR_ANALYSIS.md` has the material
   (138 zero-score questions, 52 with no relevant document in the top 100,
   per-domain breakdown) but the paper has no dedicated section with examples.
4. **Related work.** Currently two paragraphs. SemEval reviewers expect
   coverage of: multi-stage retrieval fusion (HLATR), joint document selection
   for coverage (JPR), cross-encoder distillation, and temporal IR. The
   "novelty" claim must be scoped to "empirical study of where the gain goes",
   not a new fusion method.
5. **Cost table.** The efficiency claim ("commodity GPU") needs hardware,
   wall-clock and candidate counts per stage - already measured
   (`results/`, `logs/`), not yet in the paper.
6. **One more experiment** (running): fused-metric checkpoint selection vs
   CE-only selection, on two arms' saved epochs. This converts the selection
   claim from "inferred from six arms" to "demonstrated on two arms".

## 4. Order of work

1. Fused-val selection experiment → §5.8 paragraph (or a new §5.9).
2. Error-analysis section with three worked examples (one per failure mode).
3. Related work expansion.
4. Trim to page budget, then render a real PDF (the file name "4PAGE" has
   never been validated against a template; LaTeX conversion is the honest
   check).
5. Repository sync (code + scores + TREC + qrels pointers) - already close;
   add the new probe scripts and the val-selection results.
6. Freeze a `v0.9-submission` tag, then wait for the 2027-01 window to produce
   test predictions.

## 5. Claims that must not appear

* "published" / "accepted" before the corresponding notification.
* "state of the art" - never measured against other teams.
* "the model reads only the title" - only the cross-encoder does; the dense
  stage reads the full question (§5.3 already says this).
* "step recall 96.3%" without the depth-1000 qualifier (§5.2).
* "annotation-free ceiling" - it is a fixed year-consistency rule that loses.
