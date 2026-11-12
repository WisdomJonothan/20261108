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

## 2. Page budget (confirmed against semeval.github.io/paper-requirements.html, 2026-11-13)

Authoritative rules for a single-task system paper: **5 pages for review**
(references, acknowledgments and appendices excluded), **6 pages camera-ready**
(+1 for reviewer responses). Papers are **not anonymous**: real names and
affiliations go on the title block. The title must follow
"Team Name at SemEval-YYYY Task N: Descriptive Title" (at, hyphen, colon, space
after the colon). The task description paper must be cited (a placeholder is
allowed for the initial submission; here `[abdallah2026reteco]` is the task
proposal, with the TEMPO paper as the temporary stand-in the organisers ask for).

Measured state (2026-11-13): **body ends inside page 5** (conclusion at the top
of page 5), references run from page 5 to page 6, Appendix A on pages 6-8; 0
overfull boxes, 15 references, no unresolved citations. The draft therefore fits
the 5-page review limit with the whole ladder in an appendix.

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
3. ~~**Error analysis section.**~~ Done 2026-11-13: `PAPER_4PAGE.md` §8 with
   three worked examples (pool miss, pool hit but ranked out, entity kept and
   comparison lost) and the per-domain breakdown.
4. **Related work.** Currently two paragraphs. SemEval reviewers expect
   coverage of: multi-stage retrieval fusion (HLATR), joint document selection
   for coverage (JPR), cross-encoder distillation, and temporal IR. The
   "novelty" claim must be scoped to "empirical study of where the gain goes",
   not a new fusion method.
5. **Cost table.** The efficiency claim ("commodity GPU") needs hardware,
   wall-clock and candidate counts per stage - already measured
   (`results/`, `logs/`), not yet in the paper.
6. ~~**One more experiment.**~~ Done 2026-11-13 (`results/VAL_EPOCH_SELECT.json`,
   `results/boot_dom_lx_ep1_vs_lxpossum.json`): the cross-model inversion
   replicates on 45 held-out queries, but at the epoch level both criteria pick
   the same checkpoint and the deployment-budget pick is 0.0042 *worse* on dev
   (0.3428 against 0.3470, CI [-0.0114,+0.0022]). The selection claim is
   therefore scoped to "which model to ship", not "which epoch", and the paper
   says so in §5.8.

## 4. Order of work

1. ~~Fused-val selection experiment → §5.8 paragraph.~~ Done; see 3.6 above.
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

## 7. Weight release plan (decided 2026-11-13, not yet executed)

Six checkpoints exist, 1,082 MB each: `v2m3_pool8_e2/best` (frozen, md5
`ff2b4ab8`), `lx_possum` (ep2, `64da2308`), `cx_qv` (ep1, `2b9df379`), `cx_bf`
(ep2, `1afa779c`), `sx_l03` (ep2, `64396bd2`), `sx_l10` (ep2, `a7801545`). They
are not in the repository and the submission currently cannot be rebuilt
end-to-end from public artefacts, which is the largest reproducibility gap
(`results/CHECKPOINTS.md`).

Plan: publish the frozen `v2m3_pool8_e2/best` first (one file, ~1.1 GB) as a
GitHub release asset or a Hugging Face / ModelScope model repo under the
author's account, with the md5 in the release note, a `MODEL_MANIFEST.json`
(base model, LoRA-free full fine-tune, max length 320/510, 8 frozen layers,
seed 17, 2 epochs), and a one-command rerank snippet. The five experimental
checkpoints follow only if reviewers ask; they are needed to reproduce S5.5-S5.8
exactly but every number there can also be recomputed from the committed score
JSONs and TREC runs, which is why they are second in line. Nothing here requires
new computation, only storage and credentials; until it is done the status of
the paper stays `draft` and the README says the system cannot be rebuilt from
scratch.

