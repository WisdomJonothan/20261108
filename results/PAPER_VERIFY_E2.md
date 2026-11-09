# PAPER_VERIFY_E2 — verification of `docs/PAPER_DRAFT.md` against the frozen 2-epoch arm (2026-11-09)

Scope: every numeric claim in `docs/PAPER_DRAFT.md` concerning the new headline system (dense mxbai top-100 + 2-epoch SFT bge-reranker-v2-m3, z-normalised 1:1 fusion) and leftovers of the 1-epoch arm. Values were read from frozen JSON/MD artifacts under `results/`; CE-alone 2-epoch is the only figure without a frozen JSON and was re-scored in memory from `runs/rerank_sft_e2bc_*_dev.trec` (nothing written); capture was recomputed from `results/capture_curve.json`. No other file was modified.
Verdict: both headline numbers and both CIs are correct; 8 numeric leftovers of the 1-epoch arm were found and corrected (edits below); 4 prose/label issues in the draft are flagged but not edited (plus out-of-scope artifact notes). Table 2's "frozen" label is on the correct (2-epoch) row.

## A. Abstract and headline (L19–34)

| claim (line) | artifact | read / recomputed | match | action |
|---|---|---|---|---|
| 0.3581 (L20, L209) | results/aggregate_sub_sft_e2bc_dev.json | 0.35809852326897723 | yes | — |
| dense 0.2943 (L20, L166, L203, L250) | results/phase8_mxbai_summary.json `macro.dense` | 0.2942754169514171 | yes | — |
| BM25 0.1147 (L21, L202) | github_repo/README.md:61 | 0.1147 | yes | — |
| +0.0332 (L24, L220) | results/boot_dom_e2_vs_zs.json `delta` | 0.033191131541383145 | yes | — |
| query CI [+0.0161,+0.0523] (L24) | results/boot_dom_e2_vs_zs.json | 0.0160752 – 0.0522831 | yes | — |
| domain CI [+0.0192,+0.0472] (L25) | results/boot_dom_e2_vs_zs.json | 0.0192446 – 0.0472311 | yes | — |
| "the only single-factor effect … whose domain-clustered interval excludes zero" (L25–26) | Table 3 L221 (1-epoch row) | 1-epoch domain CI [+0.0037,+0.0142] also excludes zero | no (as written) | flag — prose, see end |
| 1-epoch +0.0087 (L28, L62, L221, L227, L297) | results/boot_dom_sft_vs_zs_11.json | 0.008656796406185663 | yes | — |

## B. Table 2 (L200–216): 8 rows, 16 cells

| claim (line) | artifact | read / recomputed | match | action |
|---|---|---|---|---|
| BM25 0.1147 / — (L202) | github_repo/README.md:61 | 0.1147; topic not reported | yes | — |
| dense 0.2943 / 0.2891 (L203) | phase8_mxbai_summary.json; docs/RESEARCH_LOG.md:502 | 0.2942754; topic 0.2891 (no topic field in JSON) | yes (read) | — |
| zero-shot CE 0.2680 / 0.2990 (L204) | results/aggregate_zs_ce_dev.json | 0.2679630 / 0.2990029 | yes | — |
| SFT CE alone 0.2742 / 0.3096 (L205) | results/aggregate_sft_ce_dev.json | 0.2742367 / 0.3096486 | yes; but this is the 1-epoch CE | flag — label, see end |
| V1 RRF 0.3166 / 0.3199 (L206) | results/aggregate_sub_1_1_dev.json | 0.3166034 / 0.3198661 | yes | — |
| z 1:1 0.3249 / 0.3260 (L207) | results/aggregate_zs_z11_dev.json | 0.3249074 / 0.3259667 | yes | — |
| SFT 1e 0.3336 / 0.3339 (L208) | results/aggregate_sub_sft11_dev.json | 0.3335642 / 0.3339299 | yes | — |
| SFT 2e (frozen) 0.3581 / 0.3644 (L209) | results/aggregate_sub_sft_e2bc_dev.json | 0.3580985 / 0.3643667 | yes; "frozen" label correctly on the 2-epoch row | — |
| dense 0.2942 vs JSON 0.2942754 (L215–216) | results/phase8_mxbai_summary.json | 0.2942754 → 0.2943 | yes | — |

## C. Table 3 (L218–243)

| claim (line) | artifact | read / recomputed | match | action |
|---|---|---|---|---|
| 2e row: +0.0332, CIs (L220) | results/boot_dom_e2_vs_zs.json | as §A | yes | — |
| 1e row: +0.0087, [+0.0025,+0.0151], [+0.0037,+0.0142] (L221) | results/boot_dom_sft_vs_zs_11.json | 0.0086568; 0.0025072–0.0150914; 0.0036749–0.0142472 | yes | — |
| z vs V1: +0.0083, [−0.0010,+0.0179], [−0.0015,+0.0188] (L222) | results/boot_dom_z11_vs_v1.json | 0.0083039; −0.0010183–0.0178800; −0.0015409–0.0187615 | yes | — |
| total 1e: +0.0170, [+0.0055,+0.0290], [+0.0052,+0.0287] (L223) | results/boot_dom_sft_vs_v1.json | 0.0169607; 0.0055189–0.0289540; 0.0051765–0.0287308 | yes | — |
| "total is exactly the sum of components (0.0083 + 0.0087 = 0.0170)" (L225–226) | the two boot JSONs + exact macros | 0.0083039 + 0.0086568 = 0.0169607 (exact) | yes | — |
| "SFT − V1 = 0.3339 − 0.3199 = +0.0141" (L242) | aggregate_sub_sft11 (topic 0.3339299273), aggregate_sub_1_1 (topic 0.3198661149) | exact diff 0.0140638 → +0.0141; displayed operands gave 0.0140 | arithmetic inconsistent | fixed: 0.33393 − 0.31987 |

## D. §4 wins/losses, per-domain deltas, robustness (L231–243)

| claim (line) | artifact | read / recomputed | match | action |
|---|---|---|---|---|
| 188 wins / 78 losses / 253 ties (L231–232) | results/boot_dom_e2_vs_zs.json | `better`=188, `worse`=78, 519−188−78=253 | yes | — |
| 11 of 13 domains up (L232–233) | results/WEIGHT_SWEEP_E2.md (w=1) | 11 up, 2 down | yes | — |
| hsm +0.071 / law +0.076 / workplace +0.063 / cardano +0.055 / bitcoin −0.002 / quant −0.001 (L233–234) | results/WEIGHT_SWEEP_E2.md per-domain at w=1 | +0.071312 / +0.076021 / +0.063135 / +0.054749 / −0.002278 / −0.000893 | yes | — |
| 12-domain 0.3181 vs 0.2832 (L236) | results/ROBUSTNESS.md | 0.3181 / 0.2832 | yes | — |
| n≥19 0.3267 vs 0.2985 (L237) | aggregate_sub_sft_e2bc `domain_macro_n_ge_19` 0.3267086; ROBUSTNESS.md | 0.3267 / 0.2985 | yes | — |
| pre-registered bar sentence, +0.0141 (L240–243) | results/boot_dom_sft_vs_v1.json; README:275-279 | +0.0141 (SFT − V1, 1-epoch) | yes | operands fixed, see C |

## E. §5.1 (L249–267)

| claim (line) | artifact | read / recomputed | match | action |
|---|---|---|---|---|
| CE alone 0.2742 (1e) (L249) | results/aggregate_sft_ce_dev.json | 0.2742367 | yes | — |
| CE alone 0.2823 (2e) (L249) | re-scored runs/rerank_sft_e2bc_*_dev.trec in memory (13 domains) | 0.28229582221253624 | yes (no frozen JSON) | — |
| Spearman 0.328 (1e) (L257) | results/weight_sweep_dev.json | 0.32756145253710744 | yes | — |
| Spearman 0.279 (2e) (L257–258) | results/weight_sweep_e2_dev.json | 0.2787572253721858 | yes | — |
| oracle 0.7446 (L263) | results/capture_curve.json | 0.7446226123514619 | yes | — |
| capture 14.2% (L264) | recomputed (0.3580985−0.2942754)/(0.7446226−0.2942754) | 0.1417197819137429 → 14.2% | yes | — |
| 26.6% zero, 37.7% of zeros no gold in top-100 (L266, L87) | results/score_sub_sft_e2bc_*_dev.json (519 q) | 138/519 = 26.59%; 52/138 = 37.68% | yes | — |

## F. §5.2 (L269–305)

| claim (line) | artifact | read / recomputed | match | action |
|---|---|---|---|---|
| Table 4, all 9 rows (w, zero-shot, SFT, Δ, CI, domains up) (L273–281) | results/weight_sweep_dev.json (`bootstrap`=2000) | every cell matches after rounding (e.g. w=1: 0.3249/0.3336/+0.0087/[+0.0038,+0.0142]/10) | yes | — |
| "ahead at every w ≥ 0.85, only w = 0.7 negative (−0.0007)" (L289–290) | weight_sweep_dev.json | true (deltas +0.0004, −0.0007, then all ≥ +0.0047) | yes | — |
| "flat to within about ±0.008" (L291) | weight_sweep_dev.json | max abs(Δ) reaches 0.0149 (w=1.8) | imprecise | flag — prose |
| zero-shot arm moves 0.3192–0.3294 (L292) | weight_sweep_dev.json | min 0.3192 (1.8), max 0.3294 (0.7) | yes | — |
| 2e sweep: +0.0089 @0.5, +0.0332 @1, +0.0339 @1.5, +0.0346 @1.8, all nine positive, 13/13 at 0.7/1.5/1.8 (L301–303) | results/weight_sweep_e2_dev.json | 0.0089/0.0332/0.0339/0.0346; all 9 deltas > 0; domains_up=13 at 0.7, 1.5, 1.8 | yes | — |
| 2e sweep "is flatter" (L300) | weight_sweep_e2_dev.json | 2e spread 0.0089–0.0346 > 1e spread 0.0004–0.0149; true only for w ≥ 0.85 (0.0280–0.0346) | no (overall) | flag — prose |
| Table 4 caption describes a 1-epoch sweep and now says so (L283–286) | results/WEIGHT_SWEEP.md / weight_sweep_dev.json; e2 file is separate | yes, 1-epoch source | yes | label added |

## G. Cross-cutting leftovers (frozen system, zeros, single seed)

| claim (line) | artifact | read / recomputed | match | action |
|---|---|---|---|---|
| "27.1% of dev queries receive exactly zero (26.6% under the frozen 2-epoch system)" (L87) | results/DEV_ERROR_ANALYSIS.md:7; score_sub_* JSONs | 144/519 = 27.75% (1e); 138/519 = 26.59% (2e); no artifact yields 141/519 = 27.1% | no | fixed: 27.1% → 27.7% |
| "27.7% of queries still score zero on merit" (L439) | results/score_sub_sft_e2bc_*_dev.json | 138/519 = 26.59% | no (stale 1e value) | fixed: 27.7% → 26.6% |
| "describe +0.0087 as small and dev-specific" (L421) | results/boot_dom_e2_vs_zs.json | frozen arm's gain +0.0332 | no (old arm) | fixed: +0.0087 → +0.0332 |
| iota 0.8344 (L456) | results/ROBUSTNESS.md, 2-epoch iota row | 0.8383 | no (1e value) | fixed: 0.8344 → 0.8383 |
| per-domain range +0.0336 (cardano) to −0.0080 (bitcoin), WEIGHT_SWEEP.md (L457–458) | results/WEIGHT_SWEEP_E2.md per-domain at w=1 | +0.0760 (law) to −0.0023 (bitcoin) | no (old arm) | fixed, citation → WEIGHT_SWEEP_E2.md |
| every "frozen system" mention (25 hits) | aggregate_sub_sft_e2bc_dev.json vs aggregate_sub_sft11_dev.json | only the 1e rows (L208, L276) carry 0.3336; L192 "schedule is the frozen system" = 2e | yes | L276 relabelled "frozen weight" |
| single-seed limitation (L418–422) | results/ce_ft/TRAINING_CONFIGS.md | seed 17 for both arms; no contradiction beyond the L421 stale number | yes | only L421 fixed |

## Edits applied to docs/PAPER_DRAFT.md (8; all in-line, line numbers unchanged)

1. L87: `27.1% of dev queries receive a system score of exactly zero` → `27.7% …` — proof: results/DEV_ERROR_ANALYSIS.md:7 (144/519 = 27.75%).
2. L242: `SFT − V1 = 0.3339 − 0.3199 = +0.0141` → `SFT − V1 = 0.33393 − 0.31987 = +0.0141` — proof: aggregate_sub_sft11_dev.json topic_macro 0.3339299273296179, aggregate_sub_1_1_dev.json topic_macro 0.31986611485235017 (diff 0.014063812).
3. L276: `**1.0 (frozen)**` → `**1.0 (frozen weight)**` — proof: results/WEIGHT_SWEEP.md (1-epoch sweep; weight frozen, system not the frozen 2-epoch arm).
4. L283: Table 4 caption `Diagnostic sweep of the fusion weight on dev;` → `… on dev for the 1-epoch arm;` — proof: weight_sweep_dev.json (1e) vs separate weight_sweep_e2_dev.json.
5. L421: `We therefore describe +0.0087 as small` → `… +0.0332 …` — proof: results/boot_dom_e2_vs_zs.json `delta` 0.0331911.
6. L439: `but 27.7% of queries still score zero on merit` → `but 26.6% …` — proof: results/score_sub_sft_e2bc_*_dev.json (138/519).
7. L456: `iota's system score is 0.8344 in isolation` → `0.8383` — proof: results/ROBUSTNESS.md 2-epoch iota.
8. L457–458: `+0.0336 (cardano) to −0.0080 (bitcoin) <!-- results/WEIGHT_SWEEP.md -->` → `+0.0760 (law) to −0.0023 (bitcoin) <!-- results/WEIGHT_SWEEP_E2.md -->` — proof: results/WEIGHT_SWEEP_E2.md per-domain at w=1.

## Could not verify / flagged, not edited

1. L25–26 "the only single-factor effect … whose domain-clustered interval excludes zero": collides with Table 3 L221, whose 1-epoch domain CI [+0.0037,+0.0142] also excludes zero; true only if both SFT arms count as one effect. Suggested rewrite: "the largest single-factor effect (both arms' domain-clustered intervals exclude zero)". Prose, so not edited.
2. L205 "SFT CE alone 0.2742" is the 1-epoch CE; the frozen arm's CE-alone 0.2823 appears only in §5.1 and is missing from the Table 2 ladder. A label "(1 epoch)" or a 9th row is structural, not a numeric correction.
3. L291 "flat to within about ±0.008": |Δ| reaches 0.0149 at w=1.8; the claim holds for 8 of 9 points only.
4. L300 "flatter": 2-epoch spread (0.0089–0.0346) is larger than the 1-epoch spread (0.0004–0.0149); true only for w ≥ 0.85.
5. `results/CAPTURE_CURVE.md` is stale (1-epoch: system 0.3336, capture 8.7%). The draft's 14.2% is correct by recomputation, but the artifact needs regeneration (outside the allowed edit scope).
6. `github_repo/README.md` was updated concurrently and now matches the new arm (0.3581 / +0.0332 / 188-78-253 read against boot_dom_e2_vs_zs.json); only its rounded-operand sentence "0.3339 − 0.3199 = +0.0141" (line 281) remains, outside the allowed edit scope.
7. The original "27.1%" matches no artifact (observed zero counts: 138 / 144 / 151 / 155 / 172 / 177 of 519); correction validated against DEV_ERROR_ANALYSIS.md only.
8. Out of new-arm scope and not re-derived: §6 runtime numbers (3,086 s, 240,554 windows, 1,308 MB) and the temporal probes in §5.3–5.4; the abstract's temporal probe (+0.0039, 152 queries, 7 domains) was read and matches results/TEMPORAL_GATE.md.
