## sx_l10 vs sft_e2bc

| | dev domain-macro nDCG@10 |
|---|---:|
| sft_e2bc (baseline) | 0.3581 |
| sx_l10 | 0.3438 |
| delta | -0.0143 |

paired bootstrap (`sx_l10 vs frozen`): delta -0.0143, query CI [-0.0283,-0.0001], domain CI [-0.0351,+0.0037], 138 better / 158 worse

| mechanism | sft_e2bc | sx_l10 | delta |
|---|---:|---:|---:|
| step_cov | 0.5945 | 0.5600 | -0.0345 |
| step_recall | 0.9633 | 0.9633 | +0.0000 |
| gold_mean | 1.2103 | 1.1517 | -0.0586 |
| multi_at_k | 0.3831 | 0.3788 | -0.4300pp |

### step partition (two-step rows are the pre-registered check)

| partition | queries | domains | A (baseline) | B (candidate) | delta | 95% CI (domain-clustered) |
|---|---:|---:|---:|---:|---:|---|
| 1-step | 75 | 12 | 0.3896 | 0.3764 | -0.0132 | [-0.0411, +0.0182] |
| 2-step | 151 | 13 | 0.4058 | 0.3889 | -0.0169 | [-0.0387, +0.0040] |
| 3-step+ | 258 | 13 | 0.3511 | 0.3345 | -0.0166 | [-0.0415, +0.0077] |
| no-steps | 35 | 10 | 0.1705 | 0.1675 | -0.0030 | [-0.0585, +0.0147] |

### verdict against RESEARCH_PLAN_NEXT 4b

* main criterion (`+0.005`, CI excluding 0): **flat**
* mechanism criterion (step_cov and multi_at_10 move): **not moved**
* per the pre-registration a flat or negative main criterion is reported as *not established*, whatever the mechanism table shows

