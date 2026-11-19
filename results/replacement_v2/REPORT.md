# Bounded replacement study — results (dev, 519 queries)

## 0. Frozen baseline

* re-scored domain-macro nDCG@10 = `0.3580985233` (recorded `0.3580985233`), identical = **True**
* coverage: 13 domains, 519 queries, missing from runs = 0
* hashes: `results/FROZEN_BASELINE_20261120.json`

## 1. Frozen candidate protocol and its ceiling

* union(dense top20, CE top20) minus frozen top10; order by best rank (tie: dense before CE, then docid); keep top 3
* each candidate vs frozen ranks 8,9,10 (<= 9 pairs)
* at most one swap per query, other ranks frozen
* queries 519, pairs 4671, queries with 3 candidates 519

| | value |
|---|---:|
| baseline | 0.3581 |
| oracle (<=1 swap, frozen pairs) | 0.3769 |
| ceiling gain | +0.0188 |
| queries with a beneficial pair | 68 / 519 |
| beneficial / harmful pairs | 209 / 257 of 4671 |

> The earlier 0.4098 figure came from an *uncapped* candidate pool and is not the ceiling of this protocol.

## 2. Direct vs Context (Qwen3-4B-Instruct-2507, threshold 0, one swap/query)

| arm | macro nDCG@10 | delta | 95% CI | helpful | harmful | neutral | no swap | prompts | prompt tok | seconds | peak MB |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active | 0.3626 | +0.0045 | [+0.0004, +0.0093] | 19 | 11 | 168 | 321 | 2,630 | 1,519,002 | 249 | 8164 |
| direct | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 32 | 19 | 273 | 195 | 9,342 | 5,313,531 | 873 | 11162 |
| contextual | 0.3632 | +0.0051 | [+0.0013, +0.0094] | 27 | 7 | 219 | 266 | 9,342 | 9,404,037 | 1519 | 12045 |
| perdoc | 0.3598 | +0.0017 | [-0.0046, +0.0078] | 35 | 18 | 412 | 54 | 6,747 | 2,778,934 | 474 | 9295 |

| arm | points gained | points lost | net | pairs agreeing | pairs eligible |
|---|---:|---:|---:|---:|---:|
| active | +2.639 | -1.396 | +1.243 | 0 / 4671 | 0 |
| direct | +4.479 | -2.316 | +2.164 | 1000 / 4671 | 1000 |
| contextual | +3.617 | -0.789 | +2.828 | 651 / 4671 | 651 |
| perdoc | +4.569 | -2.457 | +2.112 | 0 / 4671 | 0 |

### Per-domain nDCG@10

| domain | n | baseline | active | direct | contextual | perdoc |
|---|---:|---:|---:|---:|---:|---:|
| bitcoin |30 | 0.2143 | 0.2011 | 0.2058 | 0.2143 | 0.2107 |
| cardano |15 | 0.2258 | 0.2258 | 0.2352 | 0.2352 | 0.2142 |
| economics |25 | 0.2512 | 0.2473 | 0.2522 | 0.2482 | 0.2430 |
| genealogy |35 | 0.4060 | 0.4086 | 0.4060 | 0.4060 | 0.4060 |
| history |240 | 0.3918 | 0.3925 | 0.3934 | 0.3973 | 0.3968 |
| hsm |45 | 0.4651 | 0.4683 | 0.4710 | 0.4709 | 0.4742 |
| iota |3 | 0.8383 | 0.8383 | 0.8383 | 0.8383 | 0.8383 |
| law |11 | 0.4357 | 0.4622 | 0.4533 | 0.4524 | 0.4382 |
| monero |19 | 0.2069 | 0.2069 | 0.2235 | 0.2069 | 0.2069 |
| politics |45 | 0.4290 | 0.4344 | 0.4375 | 0.4375 | 0.4399 |
| quant |10 | 0.1538 | 0.1538 | 0.1661 | 0.1655 | 0.1655 |
| travel |30 | 0.2494 | 0.2701 | 0.2696 | 0.2697 | 0.2647 |
| workplace |11 | 0.3881 | 0.4049 | 0.3881 | 0.3788 | 0.3788 |

## 3. Reranker controls (score vs cost)

