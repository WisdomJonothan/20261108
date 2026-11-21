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

