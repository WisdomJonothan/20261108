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
