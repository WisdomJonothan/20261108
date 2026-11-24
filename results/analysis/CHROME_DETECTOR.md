# Deployable chrome / boilerplate detector (text only)

`scripts/chrome_detect.py` scores a document in `[0,1]` for "this is boilerplate,
navigation, or a fragment" using **only the document's own `content` text** — no
qrels, no query, no LLM, no network. numpy/sklearn are not even imported at
scoring time (the LR ceiling check lives in the eval harness).

```
python scripts/chrome_detect.py --documents data/track1_tempo/<domain>/documents.jsonl --limit 20
```

## Design

- 17 bounded `[0,1]` features, each a rate or indicator, then one sigmoid:
  `score = sigmoid(-4.2 + Σ w_f · f)`.
  - **Lexical chrome**: weighted `NAV_PHRASES` (~70 phrases: `share this` 3,
    `suggest edit` 3, `table of contents` 3, `home >` 3, `all rights reserved` 3,
    `click here` 3, `jump to` 2, `cookie` 2, ...) → `nav`; an unweighted
    strong-phrase count → `chrome_phrase`; footer phrases → `footer`.
  - **Structure**: `links` (URLs + md links per word), `html` (tag density),
    `pipes` (`|`-separated list density), `bullets`, `headlines` (share of lines
    that are heading-only / title-case short lines), `repeat` (repeated line
    prefixes), `quote` (`> ` mail-quote share).
  - **Prose-ness**: `nonprose` (non-letter share), `special`, `upper`, `digit`.
  - **Fragment**: `short` (<40 words), `fragment` (no terminal punctuation /
    cut mid-sentence / lowercase start), `selfref` ("this question", "thanks in
    advance").
- Weights: nav 2.6, chrome_phrase 2.2, short 1.6, links 1.4, pipes 1.3, html 1.2,
  footer 1.2, fragment 1.1, headlines 1.0, bullets 0.8, nonprose 0.8, selfref 0.7,
  special 0.7, repeat 0.6, quote 0.6, upper 0.4, digit 0.3.
- Face validity on the 79,248 candidate documents: 135 score ≥0.9 and 1,159 (1.5%)
  score ≥0.5. The top of the range is exactly what it should be — `Table of
  Contents … 1 Introduction 2 Transactions`, `Share this on Twitter … 3
  contributors`, `Categories: 1970s British documentary television series`,
  `| Duration | Series | |---|---|`. Smoke test: nav string → 0.999, ordinary
  paragraph → 0.085, 6-word fragment → 0.525.

## Precision against the 4B model's `DUP > 0.5` label (533 sampled above-gold pairs)

| metric | value |
|---|---|
| AUC (global, n=533, 75 pos, base rate 0.1407) | **0.6248** |
| AUC (mean within-case, 37 cases with both classes) | 0.6149 |
| average precision | 0.1914 (1.36× base) |
| precision@25 / @50 / @100 / @200 | 0.12 / 0.14 / 0.26 / 0.215 |
| PR at threshold 0.3 / 0.5 / 0.9 | 0/12, 0/9, 0/2 (precision 0.000) |
| LR on the same 17 features, fitted on 2 folds of the labels, scored on the 3rd | 0.6123 mean held-out AUC (0.51 / 0.56 / 0.77) |

**The label and the detector measure different things, and the label is the
weaker one for this purpose.** All 75 `DUP>0.5` positives were inspected: **0 of
75 contain any strict navigation phrase**, while **13 of 458 negatives (2.8%) do**
— the nav feature alone has AUC 0.43, i.e. it is *anti*-correlated with the flag.
The positives are truncated fragments (24% under 70 words, 8% under 40), copies
of the question, or ordinary answer passages (67% of them have the same model's
`ANS > 0.5`). The `P_DUP` prompt bundles "copy of the question / navigation /
fragment", and the 4B judge answered mostly the first and third. A supervised
linear model on the same features tops out at 0.61 held-out, so no document-only
detector can reproduce this flag better — but the flag is not the thing worth
demoting. Every high-scoring document the detector finds (TOC blocks, share
bars, contributor lists, wiki category lines) was *missed* by the judge, which is
why precision at threshold 0.5 is zero while the ranking gain below is positive.

## Impact on the frozen fused ranking (969 out-of-fold train queries, top-10)

Rerank each query's 100-document pool by `fused_z - λ · z(chrome_score)`, with
`z` over all 79,248 candidate documents (the fused scores are already per-query
z-units). Metric: nDCG@10 with `(2^r−1)/log2(i+2)`, binary relevance; macro =
equal-weight mean of the 13 per-domain query means. Baseline macro **0.27887**.

| λ | macro | Δ | Δ pp | 95% CI (Δ pp) | rel. in top-10 base → policy | entered / left | promoted / displaced queries |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.28426 | +0.00539 | **+0.539** | [+0.134, +1.003] | 1172 → 1178 | +18 / −12 | 17 / 11 |
| 0.50 | 0.28344 | +0.00457 | +0.457 | [−0.066, +1.037] | 1172 → 1179 | +31 / −24 | 30 / 22 |
| 1.00 | 0.27878 | −0.00009 | −0.009 | [−0.978, +0.873] | 1172 → 1181 | +51 / −42 | 46 / 34 |
| 2.00 | 0.27155 | −0.00732 | −0.732 | [−2.171, +0.640] | 1172 → 1183 | +86 / −75 | 76 / 61 |

Bootstrap: 2,000 query-level resamples, 13-domain macro recomputed per draw.

**Fold-honest selection.** λ is chosen on two folds and applied to the third
(leave-one-fold-out over {0.25, 0.5, 1.0, 2.0}):

| held fold | λ chosen | held Δ pp |
|---|---|---|
| f0 (328 q) | 0.25 | +0.377 |
| f1 (323 q) | 0.50 | +0.638 |
| f2 (318 q) | 0.25 | +0.301 |
| **pooled held-out** | — | **+0.443** (Δ = +0.00443, 95% CI [+0.00036, +0.00917]) |

Per-domain Δ at λ=0.25 (pp): workplace +3.19, cardano +1.44, quant +1.00,
economics +0.69, bitcoin +0.57, history +0.41, genealogy +0.36, politics +0.20,
iota 0.00, monero −0.01, law −0.06, hsm −0.30, travel −0.46. The gain is
concentrated in a few domains, not uniform.

Calibration of the filter's firing rate: 1.46% of pool documents, 0.75% of
top-10 slots and 0.31% of gold documents score ≥0.5; λ=0.25 pushes 43 of the 73
chrome documents out of the top-10 (73 → 30 remaining).

**Rule shape matters more than the score.** Hard threshold rules are worthless:
demote everything ≥0.9 by a 3σ penalty → Δ +0.00025 (0 relevant in, 1 out);
demote ≥0.5 by 1.5σ → Δ +0.00049; gate the linear rule at score ≥0.3 → Δ +0.00104.
The gain at λ=0.25 comes from the broad mild-chrome band, not from the extreme
TOC/share blocks, which mostly sit outside the top-10 already.

**A fragment/shortness proxy is actively harmful.** Hard-demoting every document
under 70 words (the cheap stand-in for the "fragment" part of the label) costs
**−0.01304** macro (−1.30 pp), with 71 relevant documents entering and 103
leaving the top-10 — short passages are often the gold answer.

## Where the +0.0161 audit number comes from, and why +0.0044 is the deployable figure

Demoting exactly the documents the 4B model flagged (`DUP>0.5`), inside this same
evaluator and pipeline, is worth **+0.00610** (Δ pp +0.610, 3 relevant in, 0 out) —
the audit's +0.0161 instead extrapolates the 14.1% flag rate from the stratified
sample to *every* above-gold document via its own stratified estimand, so the two
are not the same quantity. Against the numbers that share a pipeline:

- deployable detector at λ=0.25: **+0.0044** fold-honest (+0.0054 fixed-λ)
- same-pipeline oracle flag demotion: +0.0061 → the detector captures **73%**
- audit extrapolation: +0.0161 → the detector captures **28%**

## Recommendation

Apply `fused_z − 0.25 · z(chrome_score)` over the 100-document pool. Expected
official 13-domain macro nDCG@10 gain: **+0.0044** (+0.44 pp, 95% CI
[+0.0004, +0.0092]), i.e. ~28% of the audited +0.0161 and 73% of what the model
flag itself achieves in the same pipeline. It is a cheap, label-free,
interpretable filter; it is not a replacement for the audit's oracle number, and
its high-precision mode (only extreme chrome) captures almost none of the gain.

Artefacts: `results/analysis/chrome_detector.json` (features, weights, AUC/PR,
per-λ table with CIs, per-domain deltas, rule variants, construct-validity audit),
`scripts/chrome_detect.py`. Evaluation harness and cached scores are in
`D:\RETECO\work_chrome\` (`eval_chrome.py`, `boot_lambda.py`, `finalize.py`,
`chrome_scores.pkl`).
