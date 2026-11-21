| policy | nDCG@10 | delta | 95% CI | prompts | prompt tokens | s | peak MB | gain / 1k prompts | source |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| oracle (cap 3) | 0.3769 | +0.0188 | - | 0 | 0 | 0.0 | 0 | - | no judge; upper bound |
| direct (cap2) | 0.3655 | +0.0074 | [+0.0026, +0.0131] | 6,228 | 3,544,580 | 967.6 | 10,306 | +0.0012 | exp_budget_cap2.json |
| direct | 0.3646 | +0.0065 | [+0.0021, +0.0117] | 9,342 | 5,313,531 | 872.3 | 11,162 | +0.0007 | exp_direct_context_dev.json |
| contextual | 0.3632 | +0.0051 | [+0.0013, +0.0094] | 9,342 | 9,404,037 | 1,518.5 | 12,045 | +0.0005 | exp_direct_context_dev.json |
| active | 0.3626 | +0.0045 | [+0.0004, +0.0093] | 2,630 | 1,519,002 | 248.7 | 8,164 | +0.0017 | exp_active_dev.json |
| qwen3-rr:single_swap | 0.3620 | +0.0039 | [-0.0007, +0.0092] | 6,747 | 3,873,426 | 706.9 | 8,358 | +0.0006 | reranker_pool_dev.json |
| direct (cap1) | 0.3610 | +0.0029 | [-0.0003, +0.0067] | 3,114 | 1,772,026 | 308.9 | 8,881 | +0.0009 | exp_budget_cap1.json |
| perdoc | 0.3598 | +0.0017 | [-0.0046, +0.0078] | 6,747 | 2,778,934 | 474.2 | 9,295 | +0.0003 | exp_perdoc_dev.json |
| qwen3-rr:full_rerank | 0.3049 | -0.0532 | [-0.0742, -0.0321] | 6,747 | 3,873,426 | 707.7 | 8,358 | -0.0079 | reranker_pool_dev.json |
| direct (dense start) | 0.3049 | +0.0106 | [+0.0048, +0.0167] | 9,342 | 5,356,653 | 871.5 | 11,298 | +0.0011 | baseline = dense-only 0.2943 |
| direct | 0.2696 | +0.0131 | [+0.0075, +0.0190] | 9,342 | 5,356,974 | 863.8 | 11,136 | +0.0014 | exp_rr20_dev.json |
| qwen3-rr:start20 | 0.2565 | +0.0000 | [+0.0000, +0.0000] | 10,380 | 5,793,582 | 1,074.0 | 8,368 | +0.0000 | reranker_pool_dev.json |
| qwen3-rr:deep100 | 0.2213 | -0.1368 | [-0.1649, -0.1085] | 51,900 | 29,022,563 | 5,523.1 | 8,684 | -0.0026 | reranker_deep100_dev.json |

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

## Single-cache budget curve (dev, no extra inference)

Every Direct pair was judged in both orders, so the same prediction cache can be replayed under any pair ordering and any budget. Deltas are domain-macro nDCG@10 against the frozen 0.358099; prompts = 2 x judge calls.

| ordering | k=1 | k=2 | k=3 | k=6 | k=9 |
|---|---:|---:|---:|---:|---:|
| slot order (8,9,10) | -0.0007 (519 calls) | +0.0007 (1038 calls) | +0.0002 (1557 calls) | +0.0066 (3114 calls) | +0.0065 (4671 calls) |
| cheap-gap order | +0.0043 (519 calls) | +0.0035 (1038 calls) | +0.0046 (1557 calls) | +0.0059 (3114 calls) | +0.0065 (4671 calls) |
| uncertainty order (AcuRank-style) | +0.0003 (519 calls) | +0.0000 (1038 calls) | +0.0003 (1557 calls) | -0.0020 (3114 calls) | +0.0065 (4671 calls) |
| random order | +0.0017 (519 calls) | +0.0041 (1038 calls) | +0.0062 (1557 calls) | +0.0067 (3114 calls) | +0.0065 (4671 calls) |
| oracle order | +0.0075 (519 calls) | +0.0104 (1038 calls) | +0.0115 (1557 calls) | +0.0099 (3114 calls) | +0.0065 (4671 calls) |

The uncertainty ordering is the weakest policy at every budget below the full pool, and a random pair at k=3 is as good as the cheap-gap order: what the stage needs is that *some* pair is judged early, not that the gap or the uncertainty picks it. With the full pool all orderings collapse to the Direct result, and the oracle ordering bounds what pair selection could add ($+0.0075$ at one pair, $+0.0115$ at three).
