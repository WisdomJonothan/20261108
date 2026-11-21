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

9,342 actions over the 519 dev queries (18 per query, 18,684 prompts, 2090 s). Dev domain-macro falls monotonically with the budget: 2: 0.35725 (delta -0.00085), 6: 0.35654 (delta -0.00156), 12: 0.35274 (delta -0.00536), 18: 0.35239 (delta -0.00571). The judge accepts more swaps as the budget grows and the extra ones are wrong. Over all judgements the mean change of a single replacement is -0.083 at slot 1 and -0.002 at slot 10, so a wrong acceptance at the top costs an order of magnitude more than at the tail, while the judge's reliability does not improve with position. A position-aware expected-gain gate fitted on the leak-free folds (delta ~ 1/slot, margin, margin/slot; 1,721 fold actions, 154 helpful) accepts 19 swaps on dev and nets 0.000000: it removes the damage but finds no gain. The value of this stage is the precision of the small CE-sourced pool, not the breadth of the action space.

## 6e. Allocation versus judgement (matched budgets, one cache)

All 18 actions per query were judged, so each ordering is replayed at matched budgets from one cache.
Delta versus the frozen 0.3581 at k = 2 / 6 / 12 / 18:

| ordering | k=2 | k=6 | k=12 | k=18 |
| --- | --- | --- | --- | --- |
| gap (prior) | -0.0009 | -0.0016 | -0.0054 | -0.0057 |
| uncertainty (AcuRank-style) | -0.0013 | -0.0048 | -0.0057 | -0.0057 |
| random | +0.0000 | -0.0006 | -0.0054 | -0.0057 |
| expected-gain model (5-fold) | -0.0008 | -0.0008 | -0.0008 | -0.0057 |
| oracle ordering (true delta) | +0.0001 | +0.0003 | +0.0002 | -0.0057 |
| oracle decision (perfect selection) | +0.0170 | +0.0362 | +0.0496 | +0.0593 |
| random order, independent run | -0.0010 | -0.0018 | -0.0035 | -0.0109 |

The independent random-order run (18,684 prompts) is -0.0010 / -0.0018 / -0.0035 / -0.0109 at the same budgets and beats the gap order at k=12, so allocation does not order the two consistently. The model's AUC for "this action helps" is 0.532. A perfect allocation over these judgements buys at
most +0.0003, while perfect selection would buy +0.0170 at k=2. The stage loses its gain at the
judgement, not at the budget.

## 6f. Condition-focused comparison

Same pool, pairs, orders and rule; the question changes from relevance to "which passage better
satisfies the conditions the query states". 15 swaps (12 helpful, 3 harmful, 91 neutral, 413
no-swap), macro 0.362524, delta +0.004426, CI [+0.00118, +0.00851], 9,342 prompts, 5.58 M tokens,
918 s. Precision 80% versus 63% for the relevance question, but a third of the swaps. The same question on the other two starts: dense start 0.2943 -> 0.3014 (+0.0071, CI
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
