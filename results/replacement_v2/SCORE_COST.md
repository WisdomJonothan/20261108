| policy | nDCG@10 | delta | 95% CI | prompts | prompt tokens | s | peak MB | gain / 1k prompts | source |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| oracle (cap 3) | 0.3769 | +0.0188 | - | 0 | 0 | 0.0 | 0 | - | no judge; upper bound |
| direct (cap2) | 0.3655 | +0.0074 | [+0.0026, +0.0131] | 6,228 | 3,544,580 | 967.6 | 10,306 | +0.0012 | exp_budget_cap2.json |
| direct | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 9,342 | 5,313,531 | 872.3 | 11,162 | +0.0007 | exp_direct_context_dev.json |
| contextual | 0.3632 | +0.0051 | [+0.0013, +0.0094] | 9,342 | 9,404,037 | 1,518.5 | 12,045 | +0.0005 | exp_direct_context_dev.json |
| active | 0.3626 | +0.0045 | [+0.0004, +0.0093] | 2,630 | 1,519,002 | 248.7 | 8,164 | +0.0017 | exp_active_dev.json |
| direct (cap1) | 0.3610 | +0.0029 | [-0.0003, +0.0067] | 3,114 | 1,772,026 | 308.9 | 8,881 | +0.0009 | exp_budget_cap1.json |
| perdoc | 0.3598 | +0.0017 | [-0.0046, +0.0078] | 6,747 | 2,778,934 | 474.2 | 9,295 | +0.0003 | exp_perdoc_dev.json |
| qwen3-rr:single_swap | 0.3577 | -0.0004 | [-0.0046, +0.0037] | 6,747 | 3,873,426 | 707.5 | 8,358 | -0.0001 | reranker_pool_dev.json |
| qwen3-rr:full_rerank | 0.2671 | -0.0910 | [-0.1206, -0.0619] | 6,747 | 3,873,426 | 707.6 | 8,358 | -0.0135 | reranker_pool_dev.json |
| qwen3-rr:deep100 | 0.1244 | -0.2337 | [-0.2746, -0.1944] | 51,900 | 29,022,563 | 5,552.1 | 8,684 | -0.0045 | reranker_deep100_dev.json |

## Decision-rule ablations (dev, recomputed from saved per-order probabilities)

| rule | nDCG@10 | delta | 95% CI | help / harm / neutral / no-swap |
|---|---:|---:|---|---|
| full | 0.3646 | +0.0065 | [+0.00202, +0.01179] | 32 / 19 / 273 / 195 |
| no_abstain | 0.3647 | +0.0066 | [+0.00119, +0.01280] | 34 / 20 / 332 / 133 |
| no_bidir | 0.3596 | +0.0015 | [-0.00706, +0.00957] | 36 / 27 / 392 / 64 |
| neither | 0.3623 | +0.0042 | [-0.00094, +0.00953] | 35 / 26 / 427 / 31 |

## Gate comparison (dev; threshold chosen on the 242 calibration queries)

| rule | delta | 95% CI | calibration delta | judge calls |
|---|---:|---|---:|---:|
| gate_judge | +0.00651 | [+0.00206, +0.01172] | - | 2916 |
| direct_all | +0.00651 | [+0.00206, +0.01172] | -0.00139 | 2916 |
| fixed_margin_0.7 | +0.00651 | [+0.00206, +0.01172] | -0.00139 | 2916 |
| fixed_margin_0.9 | +0.00620 | [+0.00182, +0.01136] | -0.00096 | 2916 |
| ce_gap_rule | +0.00492 | [+0.00229, +0.00806] | -0.00060 | 2916 |
| ce_or_dense_gap | +0.00719 | [+0.00296, +0.01231] | -0.00032 | 2916 |
| precall_cheap_gate | +0.00112 | [+0.00001, +0.00254] | - | 15.6% of pairs |

## Normalised cost per query (same 519 dev queries)

| system | nDCG@10 | delta | judge prompts / query | candidates / query | prompt tokens / query | s / query | peak MB | judge |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| direct (cap 2) | 0.3655 | +0.0074 | 12.0 | 2 | 6,829 | 1.86 | 10,306 | Qwen3-4B-Instruct-2507 |
| direct (cap 3) | 0.3646 | +0.0065 | 18.0 | 3 | 10,238 | 1.68 | 11,161 | Qwen3-4B-Instruct-2507 |
| contextual (cap 3) | 0.3632 | +0.0051 | 18.0 | 3 | 18,119 | 2.93 | 12,044 | Qwen3-4B-Instruct-2507 |
| active PRP (<=3 pairs) | 0.3626 | +0.0045 | 5.1 | 3 | 2,926 | 0.48 | 8,164 | Qwen3-4B-Instruct-2507 |
| direct (cap 1) | 0.3610 | +0.0029 | 6.0 | 1 | 3,414 | 0.60 | 8,881 | Qwen3-4B-Instruct-2507 |
| per-document scoring | 0.3598 | +0.0017 | 13.0 | 3 | 5,354 | 0.91 | 9,295 | Qwen3-4B-Instruct-2507 |
| frozen system | 0.3581 | +0.0000 | 0.0 | 0 | 0 | 0.00 | - | - |
| reranker single swap | 0.3577 | -0.0004 | 13.0 | 3 | 7,463 | 1.36 | 8,357 | Qwen3-Reranker-4B |
| dense start, direct | 0.3049 | +0.0106 | 18.0 | 3 | 10,321 | 1.68 | 11,297 | Qwen3-4B-Instruct-2507 |
| reranker full 13 | 0.2671 | -0.0910 | 13.0 | 3 | 7,463 | 1.36 | 8,357 | Qwen3-Reranker-4B |
| reranker dense-100 | 0.1244 | -0.2337 | 100.0 | 100 | 55,920 | 10.70 | 8,683 | Qwen3-Reranker-4B |

## Budget curve replayed from one prediction cache (2026-11-20)

The Direct run judged all 4,671 pairs, so a budgeted policy can be simulated
without new inference: judge only the first k pairs of a query in a given
order, apply the frozen rule to those, and measure. Every arm reads the same
stored probabilities, so differences are pair-selection differences, not
sampling noise. `scripts/budget_from_cache.py` -> `budget_cache_dev.json`.

| order | k pairs | prompts/query | nDCG@10 | delta | 95% CI | judge calls |
|---|---:|---:|---:|---:|---|---:|
| fixed | 1 | 1 | 0.3574 | -0.0007 | [-0.0079, +0.0056] | 519 |
| fixed | 2 | 2 | 0.3588 | +0.0007 | [-0.0070, +0.0081] | 1038 |
| fixed | 3 | 3 | 0.3583 | +0.0002 | [-0.0078, +0.0077] | 1557 |
| fixed | 6 | 6 | 0.3647 | +0.0066 | [+0.0012, +0.0126] | 3114 |
| fixed | 9 | 9 | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 4671 |
| cheapgap | 1 | 1 | 0.3624 | +0.0043 | [+0.0010, +0.0086] | 519 |
| cheapgap | 2 | 2 | 0.3616 | +0.0035 | [-0.0002, +0.0079] | 1038 |
| cheapgap | 3 | 3 | 0.3627 | +0.0046 | [+0.0005, +0.0094] | 1557 |
| cheapgap | 6 | 6 | 0.3640 | +0.0059 | [+0.0015, +0.0111] | 3114 |
| cheapgap | 9 | 9 | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 4671 |
| random | 1 | 1 | 0.3618 | +0.0037 | [+0.0006, +0.0076] | 519 |
| random | 2 | 2 | 0.3628 | +0.0047 | [+0.0012, +0.0090] | 1038 |
| random | 3 | 3 | 0.3641 | +0.0060 | [+0.0010, +0.0119] | 1557 |
| random | 6 | 6 | 0.3648 | +0.0067 | [+0.0023, +0.0119] | 3114 |
| random | 9 | 9 | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 4671 |
| oracle | 1 | 1 | 0.3656 | +0.0075 | [+0.0036, +0.0124] | 519 |
| oracle | 2 | 2 | 0.3685 | +0.0104 | [+0.0056, +0.0161] | 1038 |
| oracle | 3 | 3 | 0.3696 | +0.0115 | [+0.0065, +0.0173] | 1557 |
| oracle | 6 | 6 | 0.3680 | +0.0099 | [+0.0052, +0.0155] | 3114 |
| oracle | 9 | 9 | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 4671 |

Reading: at k=9 every ordering returns the frozen Direct result (0.364609, +0.006510),
which is the consistency check. At one judged pair (2 prompts/query) the cheap-gap
order keeps +0.0044 [+0.0011, +0.0086] while slot order loses and a random pair gains
+0.0037; at k=3 cheap-gap +0.0046 [+0.0005, +0.0094] against random +0.0060
[+0.0010, +0.0119]. The gap ordering therefore beats judging slot 8 first, but it does
not beat a random pair at small budgets: the value comes from judging *some* pair early
and from the bidirectional agreement rule, not from which pair the cheap gap picks.
The oracle ordering shows what perfect pair selection would buy at each budget
(+0.0075 at k=1, +0.0115 at k=3).
