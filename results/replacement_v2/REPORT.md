# Bounded replacement study — results (dev, 519 queries)

All numbers below come from committed artefacts under `results/replacement_v2/`; the frozen protocol is `configs/replacement_study_20261120.json`.

## 0. Frozen baseline

* re-scored domain-macro nDCG@10 = `0.3580985233` (recorded delta +0.0e+00, identical = True)
* topic-macro = `0.3643666957`; domains = 13, queries = 519
* hashes in `results/FROZEN_BASELINE_20261120.json`

## 1. Frozen candidate protocol and its ceiling

* union(dense top20, full-CE top20) minus frozen top10; order by best rank (tie: dense before CE, then docid); keep 3
* each candidate vs frozen ranks 8, 9, 10 (<= 9 pairs); at most one swap per query
* 519 queries, 4671 pairs

| | value |
|---|---:|
| baseline | 0.3581 |
| oracle (<=1 swap, frozen pairs) | 0.3769 |
| ceiling gain | +0.0188 |
| queries with a beneficial pair | 68 / 519 |

> The earlier 0.4098 figure came from an *uncapped* pool and is not the ceiling of this protocol.

## 2. Arms (Qwen3-4B-Instruct-2507, threshold 0, one swap per query)

| arm | nDCG@10 | delta | 95% CI | helpful | harmful | neutral | no swap | prompts | tokens | s | peak MB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0.3581 | - | - | - | - | - | - | 0 | 0 | 0 | 0 |
| direct | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 32 | 19 | 273 | 195 | 9,342 | 5,313,531 | 872 | 11162 |
| contextual | 0.3632 | +0.0051 | [+0.0013, +0.0094] | 27 | 7 | 219 | 266 | 9,342 | 9,404,037 | 1518 | 12045 |
| perdoc | 0.3598 | +0.0017 | [-0.0046, +0.0078] | 35 | 18 | 412 | 54 | 6,747 | 2,778,934 | 474 | 9295 |
| active | 0.3626 | +0.0045 | [+0.0004, +0.0093] | 19 | 11 | 168 | 321 | 2,630 | 1,519,002 | 249 | 8164 |
| direct, budget cap 1 | 0.3610 | +0.0029 | [-0.0003, +0.0067] | 15 | 11 | 157 | 336 | 3,114 | 1,772,026 | 309 | 8881 |
| direct, budget cap 2 | 0.3655 | +0.0074 | [+0.0026, +0.0131] | 26 | 13 | 224 | 256 | 6,228 | 3,544,580 | 968 | 10306 |

Context is +0.0051 with 1.77x the tokens and 1.74x the time of Direct, and the paired Direct-Context difference is -0.0015 (CI [-0.0051, +0.0016]): retained context is not a contribution.

Points gained by helpful swaps: +4.48; lost by harmful swaps: -2.32 (net +2.16).

### Per-domain nDCG@10 (frozen / direct / contextual)

| domain | frozen | direct | contextual |
|---|---:|---:|---:|
| bitcoin | 0.2143 | 0.2058 | 0.2143 |
| cardano | 0.2258 | 0.2352 | 0.2352 |
| economics | 0.2512 | 0.2522 | 0.2482 |
| genealogy | 0.4060 | 0.4060 | 0.4060 |
| history | 0.3918 | 0.3934 | 0.3973 |
| hsm | 0.4651 | 0.4710 | 0.4709 |
| iota | 0.8383 | 0.8383 | 0.8383 |
| law | 0.4357 | 0.4533 | 0.4524 |
| monero | 0.2069 | 0.2235 | 0.2069 |
| politics | 0.4290 | 0.4375 | 0.4375 |
| quant | 0.1538 | 0.1661 | 0.1655 |
| travel | 0.2494 | 0.2696 | 0.2697 |
| workplace | 0.3881 | 0.3881 | 0.3788 |

## 3. Starting systems (same candidates, judge and one-swap rule)

| start | nDCG@10 (start) | + correction | delta | 95% CI | help / harm | prompts | s |
|---|---:|---:|---:|---|---:|---:|---:|
| fused (frozen top-10) | 0.3581 | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 32 / 19 | 9,342 | 872 |
| dense top-100 | 0.2943 | 0.3049 | +0.0106 | [+0.0048, +0.0167] | 61 / 15 | 9,342 | 871 |

The correction is not specific to the fused start: it gains most where the
top ten has more clearly wrong slots.

## 3b. Query reasoning as a query transformation

The same 4B judge wrote an explicit analysis for every dev query (entities, time or version conditions, exclusions, evidence type, then search keywords; 519 queries, 499 s). With the frozen dense encoder, domain-macro nDCG@10 is 0.2933 for the original query, 0.2930 with the analysis appended, 0.2546 for the analysis alone and 0.2915 for the short query alone. Appending the analysis is neutral (-0.0003); the analysis alone loses four points; dropping the instruction costs 0.002. The published query-reasoning lift uses GPT-4o analyses with ReasonIR embeddings; at 4B scale with this encoder the transformation adds nothing.

## 4. Strong controls (Qwen3-Reranker-4B, official template, same passages)

| control | nDCG@10 | delta | 95% CI | docs scored | tokens | s |
|---|---:|---:|---|---:|---:|---:|
| single_swap | 0.3616 | +0.0035 | [-0.0032, +0.0105] | 6,747 | 3,477,302 | 587 |
| full_rerank | 0.3173 | -0.0408 | [-0.0628, -0.0188] | 6,747 | 3,477,302 | 571 |

The suffix is kept outside the truncated passage and the passage is the same 112-word query-matched window Direct sees. Single swap reaches 0.3616 with an interval that includes zero, so the reranker neither beats the judge nor is broken; full reranking of the same thirteen documents is significantly worse. An earlier right-padded run and a prefix-reading control had made the reranker look far worse than it is.

## 5. Decision-rule ablations

| rule | nDCG@10 | delta | 95% CI | help / harm / neutral / no-swap |
|---|---:|---:|---|---|
| full | 0.3646 | +0.0065 | [+0.0020, +0.0118] | 32 / 19 / 273 / 195 |
| no_abstain | 0.3647 | +0.0066 | [+0.0012, +0.0128] | 34 / 20 / 332 / 133 |
| no_bidir | 0.3596 | +0.0015 | [-0.0071, +0.0096] | 36 / 27 / 392 / 64 |
| neither | 0.3623 | +0.0042 | [-0.0009, +0.0095] | 35 / 26 / 427 / 31 |

Bidirectional agreement is load-bearing; abstention is not.

## 6. Gate (logistic regression on 3-fold, leakage-free train features)

| rule | dev delta | 95% CI | calibration delta | judge calls |
|---|---:|---|---:|---:|
| gate_judge | +0.0065 | [+0.0021, +0.0117] | - | 2916 |
| direct_all | +0.0065 | [+0.0021, +0.0117] | -0.0014 | 2916 |
| fixed_margin_0.9 | +0.0062 | [+0.0018, +0.0114] | - | 2916 |
| ce_gap_rule | +0.0049 | [+0.0023, +0.0081] | - | 2916 |
| ce_or_dense_gap | +0.0072 | [+0.0030, +0.0123] | - | 2916 |
| precall_cheap_gate | +0.0011 | [+0.0000, +0.0025] | - | 15.6% of pairs |

Threshold chosen on the 242 calibration queries: tau = 0.0549. The gate reproduces Direct; the cheap gap rule is the strongest variant; a pre-call cheap gate buys 15.6% of the judge calls for 16% of the gain.

## 6b. Where the recoverable gain sits

Of the 4524 non-relevant slots in the 519 dev top-tens, 2497 hold a relevant document inside the dense top-100, and every query improves under its best single replacement. The recoverable gain splits 43.6% / 35.0% / 21.4% across ranks 1-3 / 4-7 / 8-10. A model-assisted label over 100 fixable slots gives temporal 41, entity scope 39, numeric 10, exclusion 4, comparison 3, none 2.

## 6d. Constraint-violation filter (absolute judgement)

13 yes/no prompts per query (6,747 prompts, 505 s on one GPU) ask whether each retained document and each frozen candidate satisfies every explicit condition in the query (time period or version, entity or jurisdiction, numeric bounds, exclusions). The verdicts collapse onto relevance: candidates outside the frozen top-ten are rated satisfying by roughly zero percent, so under the relative rule (violating = 0.15 below the query median, candidate at least 0.05 above it, gap at least 0.30) only 3 of 519 queries swap, all neutral, and the macro stays at 0.358099. Direct judgement works because it forces a comparison against the incumbent document; an absolute verdict reproduces the retriever's own bias.

## 6c. Widening the action space (top-100 candidates, any of the ten slots)

Corrected 2026-11-22. The first version of this section reported losses at every budget; the
harness that produced it tokenised prompts without the chat template used by the frozen arm and
stored the reversed-order challenger probability without swapping the option names, so the
"both orders agree" rule compared the challenger with the incumbent. After the fix the two caches
agree on the 1,686 identical comparisons (mean |delta p| 0.0130, 92.8% within 0.05, 98.9% same
side, 415 vs 416 both-order agreements), and the corrected numbers are:

| start | k=2 (4 prompts) | k=6 (12) | k=12 (24) | k=18 (36) |
| --- | --- | --- | --- | --- |
| fused 0.3581 | +0.0082 [+0.0027,+0.0153] | +0.0122 [+0.0055,+0.0203] | +0.0096 [+0.0021,+0.0182] | +0.0112 [+0.0039,+0.0197] |
| dense 0.2943 | +0.0108 [+0.0062,+0.0161] | +0.0232 [+0.0146,+0.0330] | +0.0203 [+0.0125,+0.0291] | +0.0170 [+0.0106,+0.0241] |

Each cell is 519 queries, 18,684 prompts at the full budget, one swap per query. The bounded
protocol (three candidates, ranks 8-10, nine pairs) costs 18 prompts and gains +0.0065 on the
fused start and +0.0106 on the dense start, so at 12 prompts this arm matches or roughly doubles
it, and at 4 prompts it already beats it on the fused start. Helpful/harmful at k=6: 31/7 (fused),
71/8 (dense).

## 6d2. Which pairs are asked (replay over the corrected cache)

Every policy below replays the same 18 judged actions and the same acceptance rule, so only the
choice of comparisons changes. Deltas versus the frozen 0.3581 at k = 2 / 6 / 12 / 18:

| policy | k=2 | k=6 | k=12 | k=18 |
| --- | --- | --- | --- | --- |
| gap prior (registered) | +0.0082 | +0.0122 | +0.0096 | +0.0112 |
| uncertainty (AcuRank-style) | -0.0005 | +0.0036 | +0.0064 | +0.0112 |
| random within the judged set | +0.0023 | +0.0132 | +0.0107 | +0.0089 |
| random subset of the whole space (separate run) | +0.0011 | -0.0017 | -0.0058 | -0.0036 |
| oracle ordering by true delta, rule applied | +0.0258 | +0.0212 | +0.0185 | +0.0125 |
| oracle decision (best judged action) | +0.0112 | +0.0112 | +0.0112 | +0.0112 |

Reading: the gain comes from restricting comparisons to high-gap tail slots - a random subset of
the whole action space, which changes *which* pairs are asked, is flat or negative - while the
ordering inside that pool matters little (random within the judged set matches the prior at k=6
and k=12). Uncertainty ordering is the weakest of the orderings. Perfect ordering of the same
judgements would buy +0.0258 at k=2, so a better prior over which pairs to ask, not a better
order of the same pairs, is where the remaining headroom is.

## 6f. Condition-focused comparison

Same pool, pairs, orders and rule; the question changes from relevance to "which passage better
satisfies the conditions the query states". 15 swaps (12 helpful, 3 harmful, 91 neutral, 413
no-swap), macro 0.362524, delta +0.004426, CI [+0.00118, +0.00851], 9,342 prompts, 5.58 M tokens,
918 s. Of the 15 swaps that move nDCG, 12 help; the relevance question moves nDCG in 51
of its 324 swaps and helps in 32 of those, so the narrow question is the more precise one
but fires a third as often. The same question on the other two starts: dense start 0.2943 -> 0.3014 (+0.0071, CI
[+0.0024, +0.0123], 29 helpful / 10 harmful), reranker start 0.2565 -> 0.2598 (+0.0034, CI
[-0.0003, +0.0074], 27 / 8), i.e. the precision advantage is a property of the question, not
of the fused ranking, although the relevance question still gains more on both starts.

A single prompt that asks for both judgements at once (relevance and conditions) behaves like
the relevance question: 0.362638 (+0.004539, CI [-0.000503, +0.010185]), 31 helpful / 17
harmful, so the precision of the narrow question is lost when it is merged rather than run
separately.

Combined rule
(condition swap when it fires, relevance swap otherwise): 0.366117, delta +0.008018, CI
[+0.00296, +0.01383], 54 swaps; paired improvement over the relevance arm +0.0015 (CI
[-0.0001, +0.0044]) at twice the prompt cost, i.e. not established.

Second start. Starting from the dense retriever alone, the same 18-action pool gives
+0.0026 / +0.0012 / +0.0011 / +0.0006 at k = 2/6/12/18 (18,684 prompts, 1,704 s), against +0.0106
for the bounded tail protocol on that start. Widening costs most of the gain on both systems.

## 6g. Position of the correction: the same pool against ranks 1-3

The recoverable-gain analysis puts 43.6% of the fixable gain in ranks 1-3, so we re-ran the
bounded protocol with the identical pool, the identical nine pairs per query and the identical
9,342 prompts, changing only which three slots are compared. Result: 0.353916 (-0.004182, CI
[-0.021435, +0.013675]), 23 helpful / 33 harmful / 185 neutral / 278 no-swap, points +7.941 /
-10.609 -- against 0.364609 (+0.006510, CI [+0.002064,+0.011724]) with 32 healthy / 19 harmful
for the tail window. The candidate pool is held fixed, so the difference must come from the
incumbent: a doubtful tail document is cheap to overturn, a confident top-ranked one is not.

## 6i. Two further attempts at a better judge

*Evidence-first generation.* The judge is asked to quote the deciding span before naming A/B/C.
On a 39-query stratified sample (702 prompts, 310 s): 4 swaps (2 helpful, 2 harmful), delta
-0.000638, 48/702 unparsable answers. It costs about four times the single-token question and
swaps far less often (13/39 no-swap vs 5/39 for Direct), so it was not run on the full set.

*Cascade.* Sending only the queries whose Direct margin is below 0.9 (230 of 519) to the
condition judge, and letting it override Direct there, fires 13 times -- all neutral -- so the
macro is exactly Direct's 0.364609 at 26.0 prompts per query (1.44x). The condition judge has
no signal precisely where the relevance judge is uncertain.

## 6j. Splitting one budget across the two questions (cost-neutral)

18 prompts per query are split between the relevance question and the condition question
(k pairs each, both orders, frozen rule). Dev domain-macro:

| k_rel / k_cond | nDCG@10 | delta | 95% CI | swaps | helpful | harmful |
| --- | --- | --- | --- | --- | --- | --- |
| 9 / 0 | 0.364530 | +0.006431 | [+0.00203, +0.01161] | 51 | 32 | 19 |
| 7 / 2 | 0.363129 | +0.005030 | [+0.00141, +0.00925] | 45 | 28 | 17 |
| 6 / 3 | 0.364672 | +0.006574 | [+0.00276, +0.01109] | 42 | 29 | 13 |
| 5 / 4 | 0.363621 | +0.005522 | [+0.00152, +0.01009] | 40 | 28 | 12 |
| 4 / 5 | 0.363951 | +0.005853 | [+0.00175, +0.01046] | 40 | 28 | 12 |
| 3 / 6 | 0.362681 | +0.004582 | [+0.00180, +0.00785] | 36 | 26 | 10 |

The best split (6/3) is a paired +0.000149 over the all-relevance budget (CI [-0.00484, +0.00484], 12 queries better and 6 worse), i.e. precision improves (69% vs 63% of swaps
helpful, harmful 19 -> 13) while the score does not move.

## 6k. Combining the two judges

Both judgement sets are cached over the same 4,671 pairs, so the combination rules can be
replayed without new model calls (`scripts/ensemble_rule.py`, `scripts/veto_rule.py`).

| rule | nDCG@10 | delta | 95% CI | pairs used | helpful | harmful |
| --- | --- | --- | --- | --- | --- | --- |
| relevance only (Direct) | 0.364609 | +0.006510 | [+0.00206, +0.01172] | 51 | 32 | 19 |
| both questions must agree | 0.361234 | +0.003135 | [+0.00075, +0.00611] | 13 | 11 | 2 |
| condition question vetoes (tau 0.45) | 0.361556 | +0.003457 | [+0.00099, +0.00650] | 22 | 18 | 4 |

Agreement raises swap precision to 85% but keeps only 13 informative swaps; the veto removes
harmful swaps and beneficial ones in the same proportion (32/19 -> 18/4), so neither
composition beats asking the relevance question alone.

### Held-out replication (969 gate-train queries)

The same analysis on the three gate folds, where every query is ranked by a cross-encoder
that never saw it (`results/replacement_v2/error_analysis_train.json`):

| fold | queries | wrong slots | relevant in dense top-100 | queries improvable |
| --- | --- | --- | --- | --- |
| f0 | 328 | 2,862 | 1,786 | 204 |
| f1 | 323 | 2,861 | 1,779 | 200 |
| f2 | 318 | 2,795 | 1,596 | 182 |
| total | 969 | 8,518 | 5,161 (60.6%) | 586 (60.5%) |

Band shares of the recoverable gain are 44.7% (ranks 1-3), 34.1% (4-7) and 21.2% (8-10),
matching the dev values of 43.6/35.0/21.4, so the tail bias of the protocol is not an
artefact of the evaluation split. A stratified 100-slot annotation of held-out train
predictions (`error_worksheet_train_annotated.csv`) gives temporal 50, entity scope 38,
numeric 5, exclusion 4, comparison 1, none 2, against 41/39/10/4/3/2 on dev.

## 6l. A learned expected-gain model on the bounded pool

Ridge regression of the realised delta on eight features (1/slot, judge margin, margin/slot,
dense+CE gap, gap/slot, order agreement, incoming rank, dense source), fitted on 8,721
leak-free fold pairs (365 positive; train AUC 0.66, correlation 0.13) and applied to dev:

| rule | nDCG@10 | delta | 95% CI | swaps | helpful | harmful |
| --- | --- | --- | --- | --- | --- | --- |
| fixed relevance rule (Direct) | 0.364609 | +0.006510 | [+0.00206, +0.01172] | 51 | 32 | 19 |
| learned expected gain, accept if > 0 | 0.361479 | +0.003381 | [-0.00150, +0.00881] | 54 | 31 | 23 |

The model ranks actions better than chance inside the fold pool but does not transfer to the
frozen model's pairs, and its optimal threshold is not the sign rule, so the negative is
reported rather than tuned.

## 7. Fold evidence (train split, 969 queries)

| fold | queries | nDCG@10 | delta | 95% CI |
|---|---:|---:|---:|---|
| f0 | 328 | 0.2568 | +0.0007 | [-0.0033, +0.0041] |
| f1 | 323 | 0.2828 | +0.0032 | [+0.0002, +0.0066] |
| f2 | 318 | 0.3098 | +0.0068 | [+0.0022, +0.0124] |
| fall | 242 | 0.2639 | -0.0014 | [-0.0059, +0.0028] |

The folds differ in CE, start score and split, so they measure the sensitivity of the gain rather than a fresh-split estimate: fold starts 0.2562 / 0.2796 / 0.3030 give +0.0007 / +0.0032 / +0.0068, and the 242 calibration queries (start 0.2653) give -0.0014. We report the measured dev gain and these splits without extrapolating.

## 8. Case studies

See `CASES_DEV.md`: the six largest gains and losses with query text, and the per-domain swap counts.

## 9. Transfer

| collection | topics | docs | frozen og | frozen chg | Direct delta | 95% CI | help / harm |
|---|---:|---:|---:|---:|---:|---|---|
| core17 | 20 | 19,899 | 0.2674 | 0.1761 | +0.0084 | [-0.0017, +0.0186] | 8 / 3 |
| news21 | 32 | 30,921 | 0.3401 | 0.2849 | +0.0072 | [-0.0014, +0.0161] | 12 / 6 |
| robust04 | 52 | 47,492 | 0.3092 | 0.2382 | +0.0064 | [+0.0018, +0.0114] | 13 / 3 |

Official p-MRR (mteb rank_score; negative means documents that should drop under the changed instruction stay high):

| collection | dense | CE-only | frozen |
|---|---:|---:|---:|
| core17 | -0.0260 | -0.0368 | -0.0433 |
| news21 | -0.0094 | -0.0236 | -0.0208 |
| robust04 | -0.0449 | -0.0214 | -0.0347 |

Every cell is negative and no document is promoted, so the positive nDCG deltas measure ranking quality on the changed query text, not instruction-following.

Applying the judge's swaps under the changed instructions raises graded nDCG@10 from 0.1761 to 0.1825 on Core17, 0.2849 to 0.2978 on News21 and 0.2382 to 0.2430 on Robust04, but only 1, 5 and 2 of the 15, 21 and 28 applied swaps actually remove a document that the instruction change demoted. The stage improves the ranking without explicitly targeting the changed constraint.

## 9b. Does the condition question track the instruction change?

The condition judge was run on all three FollowIR collections with the same candidates,
orders and threshold (`exp_cond_<collection>_dev.json`, binary-gain harness metric):

| collection | relevance arm | condition arm | condition swaps | demoted removed (rel / cond) |
| --- | --- | --- | --- | --- |
| Core17 | +0.0084 [+0.0024,+0.0146] | 0.0000 [-0.0050,+0.0050] | 8 (1H/1X/6N) | 1/15 vs 0/2 |
| News21 | +0.0072 [-0.0014,+0.0161] | +0.0019 [-0.0041,+0.0084] | 15 (4H/3X/8N) | 5/21 (24%) vs 2/7 (29%) |
| Robust04 | +0.0064 [+0.0018,+0.0114] | +0.0019 [-0.0007,+0.0046] | 23 (4H/1X/18N) | 2/28 (7%) vs 2/7 (29%) |

Under the changed instructions the condition arm lifts graded nDCG@10 by +0.017 (Core17,
2 swaps), +0.043 (News21, 7 swaps) and 0.000 (Robust04, 7 swaps), so it fires rarely and
does not concentrate on the documents the instruction change demoted. The constraint-
response evidence required by the acceptance criterion is therefore not met.

## 9c. Correction record (2026-11-22)

The expanded-action experiments of 2026-11-21 were produced by a harness that differed from the
frozen arm in two ways: prompts were tokenised as raw text instead of through the chat template,
and the reversed presentation order stored `p[0]`/`p[1]` under the forward option names. The
second bug inverted the both-orders agreement rule, which is why 400 genuinely helpful actions
produced only 4 accepted swaps. With both fixed, the same 18,684 prompts produce +0.0122 at 12
prompts per query instead of -0.0016, and the cache reproduces the frozen arm on every overlapping
comparison. Claims withdrawn with this correction: "widening the action space does not pay",
"allocation is not the bottleneck" (the pool is), the 10x cost ratio for a wrong swap at rank 1
(the metric ratio is log2(11) = 3.46 for the same query), the "80% precision" denominator (swaps
that do not move nDCG count as neutral), and the AUC 0.532 statement as evidence that a learned
gate cannot work (that model had three constant features).

## 9d. p-MRR for the judged arms (2026-11-22)

The earlier p-MRR table covered only the retrieval stages. The judged arms are
now scored the same way: their post-swap ranking is rebuilt from the frozen
top-ten and the cached swap decision (keyed by domain and query, so the paired
`og`/`chg` instruction conditions cannot contaminate each other), and mteb's
`rank_score` is applied to the documents the instruction change demotes.

| collection | dense | CE | frozen | Direct | condition |
| --- | --- | --- | --- | --- | --- |
| Core17 | -0.0260 | -0.0368 | -0.0433 | **-0.0055** | **-0.0051** |
| News21 | -0.0094 | -0.0236 | -0.0208 | **-0.0139** | **-0.0111** |
| Robust04 | -0.0449 | -0.0214 | -0.0347 | **-0.0121** | **-0.0105** |

Every system is still negative, so none of them follows the instruction change
properly; but the correction stage roughly halves the error on Core17
(-0.0433 to -0.0055) and on Robust04 (-0.0347 to -0.0121), and demotion MRR
rises from 0.038 to 0.116 (Core17), 0.083 to 0.139 (News21) and 0.067 to 0.129
(Robust04). The honest reading is that the stage improves instruction response
without being an instruction-understanding method, which is what the earlier
"the new module does not understand instructions" sentence got wrong.

## 9e. Capability against evidence: the 2x2 grid

Same candidates, same nine pairs, same rule, one swap; only the judge model or the
length of the passage it reads changes. Coverage-aware accounting is included
because two arms can return a similar score while making very different numbers
of swaps.

| judge | passage window | nDCG@10 | delta | swaps made | helpful | harmful | beneficial-action recall | harmful loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct | 112 words | 0.364609 | +0.006510 | 324 | 32 | 19 | 0.153 | -2.316 |
| Qwen3-4B-Instruct | 400 words | 0.365998 | +0.007900 | 331 | 36 | 10 | 0.172 | -1.283 |
| Llama-3.1-8B-Instruct | 112 words | 0.362195 | +0.004096 | 330 | 24 | 12 | 0.115 | -1.532 |
| Llama-3.1-8B-Instruct | 400 words | 0.362623 | +0.004525 | 323 | 23 | 7 | 0.110 | -1.042 |

Marginal effects, all paired on the same 519 queries: more evidence is +0.001390
(CI [-0.00082,+0.00438], 21 better / 11 worse) for the 4B judge and +0.000429 for
the 8B judge; a stronger model is -0.002414 (CI [-0.00608,+0.00086], 21/22) at 112
words and -0.003375 at 400 words. Replacing the judge with a model twice the size,
given the same short evidence, does not help; widening the evidence cuts harmful
swaps for both judges (19 to 10, 12 to 7) and cuts the summed harmful loss by
about half in both cases. Neither marginal interval excludes zero, so the honest
statement is a direction: at this evidence budget the binding constraint is what
the judge can see, not how strong the judge is.

## 9g. What the labels can and cannot measure

The official dev judgments cover 228 of the 4,671 candidate pairs (4.9 per cent),
and all 228 judged challengers are relevant; 1,434 of the 1,557 slot-8 incumbents
(92.1 per cent) are unjudged as well. Every insertion the judges make therefore
falls into one of two worlds: a document the qrels cover, where the score change
is real, or a document the qrels do not cover, where the change is by definition
counted as a loss if a judged-relevant document leaves the list.

That split explains the harm counts. Of the 51 score-changing swaps the 4B judge
makes, 31 insert a judged-relevant document and 19 insert an unjudged one while
removing a judged-relevant document (one falls in neither class). Of the 8B
judge's 36, the split is 24 and 12. Within the judged subset neither judge is ever
demonstrably wrong; all of the measured degradation comes from insertions the
official pool does not contain.

Consequences, all of which the paper states: (i) the harmful-swap counts and the
harmful loss are upper bounds on real damage, not measurements of it; (ii) arms
cannot be ranked by their harm counts alone, because a judge that abstains on
unjudged candidates looks safer without being more accurate; (iii) any claim about
judge precision needs adjudication of the unjudged insertions, which is why the
next step is a reviewed sample of exactly those cases rather than more prompt
variants.

## 9h. Reading the harmful class: 19 adjudicated cases

The harmful-swap counts above are upper bounds, so we read the documents themselves.
`scripts/dump_harm.py` selects the score-changing swaps that insert an unjudged
challenger while removing a judged-relevant incumbent; the Direct arm produces 19 such
cases. Each case was adjudicated against the question (artefact
`results/replacement_v2/harm_review_verdicts.json`, single reviewer):

| verdict | cases | what the challenger is |
| --- | --- | --- |
| better | 8 | on topic and at least as useful as the removed document |
| comparable | 4 | on topic and about as useful |
| mixed | 3 | answers a different aspect of the question |
| worse | 4 | genuinely less useful |

Two of the four genuine losses are navigational pages, a table of contents and a
navigation box, which carry no prose at all. The other two are a numeric question whose
answer the incumbent states and an off-topic drift. Thirteen of the nineteen
replacements insert a document that is on topic for the question, which is what the
label-coverage bound predicts: the qrels never judged 92.1% of the candidate pool, so a
swap that removes a judged-relevant document and inserts an unjudged but reasonable one
is scored as pure loss. The practical implication is that part of the measured harm is a
labelling artefact, and that a document-type filter would remove half of the real harm at
no cost to the judge.

## 9i. Choosing the 112 words: same budget, different windows

The 400-word arm is better than the 112-word arm, so the next question is whether the
gain comes from more text or from better-chosen text. We kept the candidate pool, the
nine pairs, the prompt, the model, the 112-word display budget and the decision rule
fixed, and changed only which 112 words the judge sees (`scripts/localized_judge.py`):
`cond` takes the window that covers the question's conditions best (content terms,
numbers and dates quoted in the question, capitalised entity words), and `two` spends
the same 56+56 words on the best condition window plus the best non-adjacent one.

| arm | what the judge sees | nDCG@10 | delta | 95% CI | helpful | harmful | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct | one 112-word overlap window | 0.3646 | +0.0065 | [+0.0021,+0.0117] | 32 | 19 | 5.31 M |
| cond | one 112-word condition window | 0.3645 | +0.0064 | [+0.0020,+0.0116] | 33 | 17 | 5.32 M |
| two | two 56-word windows | 0.3613 | +0.0032 | [-0.0027,+0.0092] | 33 | 18 | 5.33 M |
| wide400 | one 400-word window | 0.3660 | +0.0079 | [+0.0030,+0.0135] | 36 | 10 | 8.22 M |

Paired against the frozen 112-word rule: `cond` +0.00047 [-0.00064,+0.00157] and `two`
-0.00087 [-0.00381,+0.00198], i.e. no difference either way. Choosing the window by the
question's conditions does not reproduce the 400-word gain, and splitting the same budget
over two windows loses a little context. The conclusion of the evidence axis is therefore
narrow and negative: the judge benefits from more raw text, not from a better-chosen 112
words, and even the 400-word arm's paired interval against the 112-word rule includes
zero (-0.0008, +0.0044). At this point neither a stronger model nor smarter selection
moves the result reproducibly, which is the stop condition of the capability/evidence
programme; the system paper is written around the frozen Direct configuration.

## 12. Negative-label audit and facet recall (2026-11-23)

### 12.1 Does treating unlabelled documents as negatives hurt? (experiment A)

The CE is trained with `hard + random` negatives drawn from the dense top-100, and
62.4% of the dev pool documents that the system gets wrong were never judged. If
those unjudged documents were often relevant, the training signal would be
contaminated. Two retrainings test the hypothesis directly, with everything else
(supervision source, init, budget, evaluation chain) held fixed:

| negative set | dev domain-macro | delta |
| --- | --- | --- |
| frozen recipe (hard top-20 + random) | 0.3582 | - |
| hard negatives with dense rank <= 10 removed | 0.3409 | -0.0173 |
| no hard negatives (only dense rank 21-100 + random) | 0.3308 | -0.0274 |

Removing the suspicious negatives makes the CE **worse**, monotonically in how much
is removed (law -0.048, workplace -0.048, hsm -0.034 for the no-hard arm; bitcoin
+0.004 and quant +0.001 are the only gains). The unjudged hard negatives are not
noise to be cleaned away: they are where the cross-encoder learns to separate the
top of the list, and the dev pool is drawn from exactly that band.

A blind model sample of 260 hard negatives (two per query, dense ranks 1-20,
no qrels consulted during sampling) supports the *premise* while contradicting the
*remedy*: a Qwen3-4B judge assigns p(relevant) >= 0.5 to 86 of 260
(33.1%) and >= 0.9 to 76, and the rate rises
towards the head of the list (mean p = 0.658 for dense ranks 1-2, 0.241 for ranks
11-20). So the labels are indeed noisy, but the documents carrying that noise are
also the ones the model most needs to see; filtering them out is not a fix. If this
line is pursued at all it must be re-weighting rather than removal.

### 12.2 Does facet retrieval recover the documents the query misses? (experiment B)

The official train/dev splits ship decomposition steps (`steps_dev.jsonl`,
`qrels_steps_dev.txt`) and a per-step relevance file. Using each step as its own
BM25 query over the same corpus, and comparing with the whole-query BM25 retrieval:

| quantity | value |
| --- | --- |
| dev queries with steps | 484 |
| whole-query BM25 finds a judged document in the top-100 | 250 |
| queries it misses | 234 |
| of those, at least one step recovers a judged document | 113 (48%) |

Per domain the effect is concentrated where the corpus is large and the questions are
multi-hop (history 68/132,
genealogy 10/15,
travel 9/17) and absent
for politics/hsm/monero.

At the pool level, however, the dense retriever is much stronger than BM25, so the
union only adds a little: pool coverage of the 138 zero-score queries goes
86 -> 94 (+8), and the union adds
42 judged-relevant documents that dense never returns. Reranking the
union pool with the frozen cross-encoder converts 5 of the 15 queries that
gained a new judged document into a non-zero score (mean nDCG@10 0.081 over
those 15), which is worth about **+0.0015 domain-macro** at best.

Conclusion: step-level retrieval is a real but small lever. It changes pool coverage
by eight queries and the frozen reranker converts about a third of them, so it cannot
carry a score increase on its own; it is recorded here so the same ground is not
re-walked.

## 18. How much score is a corrected temporal misordering worth? (2026-11-24)

The question: if every *verified* temporal misordering in the out-of-fold rankings were corrected, how much official nDCG@10 returns? Measured on the 969 out-of-fold train queries (three folds, frozen top-100 fixed), stratified sample of 138 queries / 533 documents ranked above their gold, four order-free Qwen3-4B probes (query is temporal; passage is boilerplate; passage answers; passage is about another period), correction replayed and re-weighted to the frame, 2,000 bootstrap draws.

| rule applied to the replay | corrected pairs | macro delta | 95% CI |
|---|---:|---:|---|
| automatic screen (wrong period > 0.6, gold not answering) | 14 | +0.0075 | [+0.0010, +0.0175] |
| conservative screen (also drops boilerplate and non-answering documents) | 9 | +0.0064 | [+0.0002, +0.0165] |
| **manual audit, only genuine period/direction errors** | **5** | **+0.0026** | [+0.0003, +0.0053] |
| manual audit plus the two borderline cases | 7 | +0.0033 | [+0.0008, +0.0063] |

I read all 15 automatically flagged pairs. Five are real period or direction errors (a 2020 article above a 2017 launch date; an 1860 census above an 1840 one when the question asks about pre-1850; pre-1965 background above a document about the 1965 embargo; a fifteenth-century account above a medieval one for a "when did it begin" question; a traveller's forum question above the official rule that answers it). Two are borderline (entity/scope confusions rather than time). Eight are false positives: three are page chrome ("Share this on Twitter", "3 contributors ... Suggest edit"), two are pairs where both documents are on topic, and three are cases where the document ranked above the gold is arguably better than the gold.

So the automatic flag has 33-47% precision, and the **verified temporal ceiling is +0.0026 macro (0.7% relative), at most +0.0053 at the CI upper bound** — below what the existing six-pair judge already captures out of fold (+0.0045), and about 1/75 of the +0.2 target. Per the pre-agreed decision rule, a temporal-inversion training programme is not justified on score grounds; the frozen 0.3735 configuration stands.

The audit also ranks the actual loss classes, which is the useful part:

| class of document ranked above the gold | share of above-pairs | ceiling if fully corrected |
|---|---:|---:|
| plain topical difference / plausible but unjudged | 44.4% "answers the question" | not measurable (label incompleteness) |
| documents the model calls boilerplate/navigation/fragments (a heterogeneous flag: truncated fragments and question copies, not page chrome) | 14.1% | +0.0161 [+0.0099, +0.0228] (in-pipeline: +0.0061) |
| temporal / period misordering | 2.8% (1.0% verified) | +0.0026 [+0.0003, +0.0053] |

The largest single exploitable defect is not temporal reasoning but documents the judge calls boilerplate/navigation/fragments. A follow-up construct-validity audit corrected the label: **none of the 75 flagged documents contains a navigation phrase** — they are truncated fragments (24% under 70 words), question copies, and ordinary answers (67% also score as answering the question), so this is not page chrome and a strict chrome filter cannot capture it. A text-only detector built for the deployable version (`scripts/chrome_detect.py`, 17 lexical/structure/fragment features, no qrels and no LLM) reaches AUC 0.625 against those flags and, applied as `fused_z − 0.25·z(chrome)`, gives **+0.0044 fold-honest macro nDCG@10** (CI [+0.0004,+0.0092]; fixed lambda +0.0054), i.e. 28% of the audited ceiling and about the size of the whole replacement gain. Hard demotion thresholds and a short-document proxy do not work (+0.0003 and −0.0130 respectively). Caveat on both ceilings: the replay corrects only the sampled pairs and re-weights by stratum, so these are ideal-correction estimates over an audited sample, not exhaustive top-100 oracles; and because RETECO qrels are positive-only, "unjudged but plausible" documents cannot be separated from genuinely better ones.

## 10. Conclusion (2026-11-22)

The frozen system re-verifies at 0.3580985233 and a bounded local comparison over a three-candidate
tail pool improves it to 0.3646 (+0.0065, CI [+0.0021,+0.0117], sign-flip p is uninformative and a
shuffle test over the same pool gives p = 0.0007), 35% of that pool's 0.0188 ceiling, and improves two
other starting systems (+0.0106 dense, +0.0131 reranker start).

What does not add to that, each reported above with its interval: retained context (paired -0.0015),
the learned gate (it reproduces Direct and filters nothing at its calibration threshold), cheap-gap
allocation (inside one standard error of a random pair at k=3), absolute condition checks (three swaps,
all neutral), the condition-focused pair question on the fused start (+0.0044 alone, and its merged
"either" rule only +0.0015 over Direct at twice the prompts), query reasoning as a query
transformation (0.2933 -> 0.2930), the off-the-shelf reranker control (not better, interval includes
zero) and a stronger judge or a longer evidence window (2x2 grid: capability -0.0024 at 112 words,
evidence +0.0014, both marginal).

The one correction that did reverse is the widened action space: once the inference path was made
identical to the bounded arm (chat template, symmetric option names), 18 judged actions per query give
+0.0082 / +0.0122 / +0.0096 / +0.0112 at k = 2/6/12/18 against Direct's +0.0065 at 18 prompts, and
reordering the same judgements by their true effect would buy +0.0258 at k = 2. The paired advantage
over Direct is +0.0056 at k = 6 with CI [-0.0009,+0.0127], so it is a direction, not a result, and
Direct stays the frozen configuration.

FollowIR transfer reproduces the fusion advantage and a small Direct gain but no instruction
following (p-MRR -0.0208..-0.0433 for the frozen system, -0.0055..-0.0139 for the judged arms). The
score is low because 138 of 519 dev queries return nothing relevant in the top ten and 52 of those have
no relevant document in the delivered top-100 at all, so the line of work reported here operates on the
381 queries that already retrieve something (docs/SCORE_DIAGNOSIS_20261122.md).

Limitations that a reviewer should weigh: at least 120 dev configurations were evaluated; the
headline's CI-implied p is 0.0083, which survives a Bonferroni correction only for a family of six or
fewer; six post-freeze variants on the same start exceed it (hybrid +0.0080, wide400 +0.0079, cap2
+0.0074, ce_or_dense_gap +0.0072, split 6/3 +0.0066, widened k=6 +0.0122) and every paired interval
against Direct includes zero; the +0.005 acceptance threshold was written into the protocol file after
the first run (the pool and the ceiling were frozen before it); and the widened-action harness postdates
its artefacts, with the probability cache shipped so the decisions can be re-derived.

See docs/DECISIONS_20261121.md and results/audit/AUDIT.md.


## 12. Swapping the first-stage embedder (three-domain probe)

Qwen3-Embedding-0.6B replaced mxbai-embed-large-v1 on three dev domains (bitcoin, quant, law),
everything downstream unchanged (same frozen CE checkpoint, same z 1:1 fusion, same detie).

| domain | dense nDCG@10 mxbai -> qwen3 | R@100 mxbai -> qwen3 | fused mxbai -> qwen3 | delta |
| --- | --- | --- | --- | --- |
| bitcoin | 0.1820 -> 0.1824 | 0.5646 -> 0.5794 | 0.2143 -> 0.2199 | +0.0056 |
| quant | 0.1954 -> 0.1698 | 0.4833 -> 0.5167 | 0.1538 -> 0.1872 | +0.0334 |
| law | 0.3689 -> 0.3860 | 0.8545 -> 0.9409 | 0.4357 -> 0.4061 | -0.0295 |

Recall@100 improves in all three domains (+1.5 to +8.6 points) but the fused score is mixed:
quant gains a third of its score, law loses three points, and the mean over the three domains is
+0.0032, i.e. roughly +0.0007 on the 13-domain macro. Encoding the remaining ten domains takes
about eight hours on this contended machine, so the expected return does not justify it under the
current evidence; the probe is recorded as a boundary result (better coverage does not translate
into a better slate when the cross-encoder is frozen).

## 11. Score-side experiments (2026-11-22 evening)

Three experiments asked whether the *first stage* rather than the judge is the bottleneck.

**The frozen system is reproducible end-to-end.** Rerunning the frozen checkpoint
(`results/ce_ft/v2m3_pool8_e2/best`) through the documented chain (dense top-100, title
query capped at 256 tokens, window 250/192/3, pair-max-len 510, z 1:1 fusion) gives
**0.3581821** against the recorded **0.3580985** (delta 8.4e-5); 11 of 13 domains agree to
four decimals, economics and workplace differ by 0.0006 and 0.0017. The original CE run
files are not in the repository, which is why the earlier `fuse_scores.py` probe on
`runs/fusion100_full_ce_s17_*` could not match the submission; with the frozen checkpoint
rerun the pipeline reproduces.

**A slate-metric loss does not help the cross-encoder.** Retraining the same CE with
`--loss ndcg` (lambdarank with delta-nDCG@10 pair weights, 969 train queries, 2 epochs,
lr 1.5e-5, seed 17) and evaluating through the identical chain gives **0.3519337**, i.e.
**-0.0062** against the reproduction. Five domains improve (travel +0.0074, cardano +0.0026,
iota +0.0028, bitcoin +0.0013, quant +0.0009) and eight lose, worst monero -0.0215,
history -0.0195, hsm -0.0181, law -0.0149. The loss function is not the limiting factor at
this data scale.

**The 1:1 fusion weight cannot be improved from train.** Sweeping w in {0.5, 0.75, 1, 1.25,
1.5, 2, 2.5, 3} on the 969 train queries with the leak-free out-of-fold CE runs selects
**w = 0.75** (train macro 0.2811 vs 0.2789 at w = 1), but on dev that choice gives **0.3486**
while w = 1 gives 0.3582. The train-optimal weight transfers negatively, so the frozen 1:1
choice is kept. The dev-optimal w = 1.5 (0.3591) was observed after the fact and is not
adopted.

Together these rule out "the CE was trained for the wrong objective" and "the fusion weight
was never tuned" as explanations of the remaining gap; the gap sits in the cross-encoder's
ranking quality itself (pool oracle 0.7446 versus 0.3581 delivered).
## 13. Widened action space as the competition configuration (2026-11-22)

The corrected widened space (top-100 candidate pool, any of the ten slots, one swap) replayed at
six judged pairs per query is the strongest configuration measured on dev:

| configuration | prompts/query | nDCG@10 | delta | 95% CI |
| --- | --- | --- | --- | --- |
| frozen | 0 | 0.3581 | - | - |
| tail replacement (Direct), 9 pairs | 18 | 0.3646 | +0.0065 | [+0.0021,+0.0117] |
| widened, 6 pairs, prior order | 12 | 0.3703 | +0.0122 | [+0.0054,+0.0203] |
| widened, 6 pairs, head-first order | 12 | 0.3735 | +0.0154 | [+0.0065,+0.0256] |

Head-first ordering (rank by slot, then by the dense+CE prior) is nominally better than the prior
ordering by +0.0032, but the paired interval spans zero [-0.0015,+0.0094], so the ordering is not
established; both are reported. A ridge selector over the judged actions, cross-validated over the
519 dev queries, reaches +0.0149 [+0.0061,+0.0249], i.e. +0.0036 [-0.0034,+0.0110] over the
frozen decision rule, so learning the selection is promising but not yet established.

Cost: six judged pairs cost 12 prompts per query against 18 for the tail arm, with fewer tokens
(about 1.8 M against 5.3 M over 519 queries), so the stronger configuration is also the cheaper one.
The caveat that applies to every dev number stands: k=6 was picked after inspecting the budget
curve, and the paired advantage over the tail arm includes zero.
## 14. Does the configuration transport to the training split? (2026-11-23)

The widened space was rerun on all 969 gate-training queries (three folds, each judged only by a
cross-encoder that never saw that query: 34,884 prompts, 33 min per fold on one 3090). Judging the
first six pairs in head-first order and judging all eighteen give:

| split | queries | head-first, 6 pairs | all 18 pairs (max-mp) |
| --- | --- | --- | --- |
| fold f0 | 328 | +0.0036 | +0.0038 |
| fold f1 | 323 | +0.0065 | +0.0114 |
| fold f2 | 318 | +0.0034 | +0.0019 |
| pooled train | 969 | **+0.0045** | +0.0057 |
| dev | 519 | +0.0154 | +0.0112 |

Two honest conclusions follow. First, the dev split is favourable to this method: the pooled
training estimate is +0.0045, not +0.0154, and on training queries restricting the judge to six
pairs is not better than judging all eighteen. Second, an expected-gain selector fitted on the
training folds (ridge on merged probability, agreement, slot weight, prior and the two order
probabilities; 17,442 actions) scores +0.0042 in grouped cross-validation on training and +0.0086
on dev, i.e. it does not beat the simple rule once it is fitted on data that does not include the
evaluation queries. The earlier +0.0149 came from cross-validation inside dev and does not transport.

The submission configuration therefore stays the six-pair head-first variant (12 prompts per query,
cheaper than the eighteen-prompt tail arm) with the expectation of a **+0.005** gain on unseen
queries, not the dev number.

## 16. Review round 3 (2026-11-24): loss function, specialist reranker, judgement

Four questions from the review were answered with measurements; three are negative, and the fourth localises the remaining loss.

**16.1 The nDCG loss, with the pair mask fixed.** The mask in `train_ce.py` was `(rank_i < k) & (rank_j < k)`, which drops every pair that straddles the top-k boundary and gives zero gradient whenever no two positives sit inside the top ten together. It is now `|`. With the corrected mask, lambdarank nDCG@10 training on the 969 train groups scores **0.3514** on dev; a possum-loss control trained on exactly the same groups, initialisation and budget scores **0.3572**, while the frozen checkpoint (1,211 queries, its own group set) reproduces 0.3582. The comparison is single-factor and the slate loss is **-0.0058**: the loss function is not what limits the cross-encoder at this data size.

**16.2 A specialist group reranker on the same pool.** Diver-GroupRank-7B, run with its official template on the dense top-20 as one group (180-word excerpts, answer prefilled so the 7B does not spend the budget reasoning), scores 0.244267 against the dense order's 0.245332 over 65 queries (five per domain): **-0.0011 nDCG@10**, while moving seven more relevant documents into the top ten (55 -> 62) and repairing three zero-score queries (28 -> 25). The two-query-per-domain sample had suggested +0.08; that was a property of the sample, not of the model.

**16.3 Where the 0.3735 configuration loses.** Replaying the frozen rule over the cached 18 judged actions reproduces 0.373503. The ladder above it is: perfect selection among the actions the judge already accepted **0.380299**; perfect judgement and selection inside the six judged pairs **0.401902**; inside all eighteen **0.417443**. Of the 85 queries that have a beneficial action among the six pairs, 33 end positive, 26 reject every beneficial action and 26 accept one but pick a different action. Six alternative selection rules (prior, product, gain-weighted prior, slot-first, high-confidence prior) replayed over the same judgements reach at most 0.373503, so no selection rule recovers the gap: the accept/reject decision loses the points.

**16.4 Step-level support judgement.** 220 (query, requirement, document) triples from the train split were judged in a single prompt: does the passage contain the specific evidence this requirement asks for. Gold documents score 0.367 mean probability under their own requirement and 0.261 under a sibling requirement of the same query (negatives from dense ranks 21-100: 0.075); the same documents under the global question (does this passage help answer the question) score 0.442 against 0.094. Ranked AUC is 0.8156 for the requirement-conditioned judgement and 0.8025 for the global one, and 136 of the 220 gold documents are scored strictly higher under their own requirement than under a sibling. Requirement conditioning carries some signal, but at 4B it is worth about one AUC point over topical relevance, and the absolute level is low: the model says the gold document contains the required evidence only about a third of the time. That is consistent with 16.3 - the binding constraint is what the judge can verify, not which pair it is asked about.

### 16.2b The pool-100 cascade does not hold up either (65 queries)

The 26-query cascade (keep the best five per 20-document group, rerank the 25 survivors) gave
0.185133 -> 0.259044 (+0.074). Extended to 65 queries (5 per domain) the same setting gives
**0.245332 -> 0.232081 (-0.0133)**, zero-score queries 28 -> 25, relevant-in-top-10 55 -> 56.
Both Diver variants are therefore nDCG-neutral or worse once the query sample is not the first
two queries of each domain; what it does do, consistently, is move a handful of additional
relevant documents into the top ten. The specialist-reranker line is closed.

## 17. Does requirement-conditioned evidence help? (2026-11-24)

Question, fixed in advance: on the same six pairs, the same 4B judge and the same
total input budget, does pointing the judge at the evidence the question's own
requirements ask for make it accept more of the swaps that actually help?

Design. Four cells over the three held-out folds (754 queries that carry at least
two decomposition steps), 12 prompts per query in every cell:

| cell | passage text shown | comparison question |
|---|---|---|
| A (reference) | the 112-word query-overlap window | original relevance question |
| sel | 112-word window chosen to cover the requirement | original relevance question |
| req | the original window | which passage better provides what the requirement asks for |
| both | requirement-selected window | requirement-conditioned question |
| mis | window selected for a *sibling* requirement | question asks the sibling requirement |

`mis` is the decisive control: same machinery, same length, wrong requirement. If
the gain is about the requirement matching the evidence, `req` must beat `mis`.

Out-of-fold results (frozen baseline 0.300255 on these queries):

| cell | nDCG@10 | vs frozen | vs cell A | changed / helped / hurt |
|---|---:|---:|---:|---|
| A six pairs | **0.307869** | +0.007614 | — | — |
| sel | 0.304220 | +0.003965 [-0.000638, +0.008746] | **-0.003649 [-0.007432, -0.000266]** | 41 / 18 / 23 |
| req | 0.306841 | +0.006586 [+0.001801, +0.011832] | -0.001028 [-0.005928, +0.004083] | 43 / 18 / 25 |
| both | 0.304290 | +0.004035 [-0.000338, +0.008824] | -0.003580 [-0.008062, +0.000903] | 58 / 23 / 35 |
| mis | 0.305816 | +0.005561 [+0.001616, +0.010270] | -0.002054 [-0.006330, +0.002501] | 61 / 26 / 35 |

Matched minus mismatched requirement: **+0.001025, 95% CI [-0.003317, +0.005464],
p = 0.327**. Selection on the requirement alone is *worse* than the frozen rule by
0.0036 with an interval that excludes zero; adding the requirement question does
not recover it; the requirement question on its own is neutral.

Reading. The experiment was designed so that a positive result would have to show
up as `req` > `mis`. It does not. Whatever the cells buy comes from changing what
the judge is asked, not from the requirement being the right one, and the version
that changes the passages instead of the question loses ground. This matches the
other three measurements of the same mechanism: step-level support judgement is
worth about one AUC point over the global question (section 16.4), requirement-
bearing text is equally often outside the shown window in successful (38.6%) and
failed (30.2%) swaps, and 55% of the missed beneficial swaps are confident
refusals below p = 0.35 rather than agreement-rule vetoes (26%).

Consequence for the competition configuration: nothing here beats cell A, so the
frozen 0.3735 configuration is unchanged. Per the pre-agreed stopping rule (a
candidate must beat the six-pair rule out of fold and beat the mismatched
requirement), this line stops before any dev confirmation, before expanding six
pairs to eighteen and before allowing a second swap.

## 19. Three more nulls and one marginal variant (2026-11-24)

Three cheap levers were measured after the temporal audit, all on fixed pools and
all replayed from cached judgements where possible.

**Generative reverse scoring (UPR-style).** Each of the frozen top-100 documents was
scored by the mean token log-probability of the query given the document, using
Qwen3-4B-Instruct-2507 with a "write the question this passage answers" prompt
(400-word excerpts). Over the 969 out-of-fold queries (96,900 pairs):
frozen fused 0.278869, reverse score alone 0.062374, and the per-fold-weighted
combination 0.277983 (-0.000886). The reverse score is essentially orthogonal to the
fused score (mean per-query Spearman -0.0048) but far too weak to help: it repairs
31 of the 298 zero-score queries while losing more elsewhere, and every domain drops
when used alone (law -0.346, workplace -0.336, hsm -0.273).

**Predicting where the ranking fails.** Cheap within-query spread features of the
reverse scores (std, range, coefficient of variation, fused std, pool size) do not
separate zero-score queries from the rest: AUC 0.44 / 0.45 / 0.48, i.e. at or below
chance. There is no cheap signal for routing.

**Text-only boilerplate demotion transfers?** No. The detector that gains +0.0044 on
the out-of-fold training splits (section 18) was applied to the dev pool
(`z(fused) - lambda * z(chrome)`): 0.358108 at lambda=0, 0.357777 at 0.1, 0.357754 at
0.25, 0.355631 at 0.5. Bitcoin (+0.0125 at 0.5) and cardano (+0.0076) improve while
workplace (-0.0133), quant (-0.0302), hsm (-0.0188) and politics (-0.0144) fall, so
the training-split gain does not replicate on dev. This is the fourth lever whose
out-of-fold training gain fails to transfer.

**Acceptance margin and multi-swap.** Replaying the cached two-order judgements with
the six-pair budget: the margin threshold does not matter (every tau from 0.00 to
0.30 gives +0.0154 on dev because the judge's probabilities are saturated near 1.0),
and allowing a second or third disjoint swap gives +0.00486 / +0.00478 on the training
splits against +0.00448 for one swap, and +0.01569 / +0.01575 on dev against +0.0154.
The direction is consistent but the size is inside noise, so the frozen one-swap rule
is kept for the competition entry.

Net: the competition configuration stays at 0.3735 (dev), with the pooled training-fold
expectation of about +0.005. Four pool-level reorderings (BM25 fusion, per-domain
fusion weight, Qwen3-Embedding-0.6B, UPR reverse scoring) and two ranking-quality
filters (chrome demotion, learned gate) have now been measured and rejected.
