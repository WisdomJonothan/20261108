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

## 10. Conclusion (2026-11-21)

The frozen system re-verifies at 0.3580985233 and a bounded local comparison over a three-candidate
tail pool improves it to 0.3646 (+0.0065, CI [+0.0021,+0.0117]), 35% of that pool's 0.0188 ceiling,
and improves two other starting systems (+0.0106 dense, +0.0131 reranker start). Retained context,
a learned gate, cheap-gap allocation, a widened top-100 action space, absolute condition checks,
condition-focused pairs, query reasoning and the off-the-shelf reranker control do not add to that:
each is reported above with its interval. The widened-space failure is located at the judgement
(AUC 0.532; oracle selection +0.0593 versus oracle allocation +0.0003), which is where any further
work has to start. FollowIR transfer reproduces the fusion advantage and a small Direct gain but no
instruction following (p-MRR -0.0208..-0.0433). See docs/DECISIONS_20261121.md.
