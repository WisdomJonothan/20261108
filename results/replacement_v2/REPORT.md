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

Direct improves nine of thirteen domains, worsens one (bitcoin, -0.0085) and leaves three flat (genealogy, iota, workplace). 
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
| Qwen3-Reranker-4B on dense top-20 | 0.2565 | 0.2696 | +0.0131 | [+0.0075, +0.0190] | 57 / 14 | 9,342 | 864 |

The correction is not specific to the fused start: it gains most where the
top ten has more clearly wrong slots.

## 4. Strong controls (Qwen3-Reranker-4B, same candidates)

| control | nDCG@10 | delta | 95% CI | docs scored | tokens | s |
|---|---:|---:|---|---:|---:|---:|
| single_swap | 0.3620 | +0.0039 | [-0.0007, +0.0092] | 6,747 | 3,873,426 | 707 |
| full_rerank | 0.3049 | -0.0532 | [-0.0742, -0.0321] | 6,747 | 3,873,426 | 708 |
| deep-100 rerank | 0.2213 | -0.1368 | [-0.1649, -0.1085] | 51,900 | 29,022,563 | 5523 |

Per-document scoring with the same judge (13 yes/no prompts per query) reaches 0.3598 (+0.0017, ns); the cheap-gap pair-selection control reaches 0.3626 but its paired difference from Direct includes zero.

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

## 7. Fold evidence (train split, 969 queries)

| fold | queries | nDCG@10 | delta | 95% CI |
|---|---:|---:|---:|---|
| f0 | 328 | 0.2568 | +0.0007 | [-0.0033, +0.0041] |
| f1 | 323 | 0.2828 | +0.0032 | [+0.0002, +0.0066] |
| f2 | 318 | 0.3098 | +0.0068 | [+0.0022, +0.0124] |
| fall | 242 | 0.2639 | -0.0014 | [-0.0059, +0.0028] |

The three folds average +0.0036 and the calibration queries -0.0014, so the dev gain of +0.0065 is an upper draw; +0.002 to +0.003 is the honest expectation on a fresh split.

## 8. Case studies

See `CASES_DEV.md`: the six largest gains and losses with query text, and the per-domain swap counts.

## 9. Transfer

FollowIR Core17 (20 topics) was re-run from scratch under the frozen protocol: dense mxbai top-100, frozen CE rerank, z 1:1 fusion, then the same cap-3 replacement arms. Graded nDCG@10 of the frozen system is 0.2674 with the original instructions and 0.1761 with the changed instructions (dense 0.2256 / 0.1608; CE-only 0.2523 / 0.1516). The Direct judge moves the harness binary-gain mean from 0.3039 to 0.3123 (delta +0.0084, CI [-0.0017, +0.0186]; 8 helpful, 3 harmful, 18 neutral, 11 no-swap of 40 queries). The interval includes zero: this is transfer evidence in the same direction, not a second significant win. News21 and Robust04 are not yet run.
