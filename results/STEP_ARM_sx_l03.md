## sx_l03 vs sft_e2bc

| | dev domain-macro nDCG@10 |
|---|---:|
| sft_e2bc (baseline) | 0.3581 |
| sx_l03 | 0.3442 |
| delta | -0.0139 |

paired bootstrap (`sx_l03 vs frozen`): delta -0.0139, query CI [-0.0283,+0.0006], domain CI [-0.0348,+0.0044], 142 better / 158 worse

| mechanism | sft_e2bc | sx_l03 | delta |
|---|---:|---:|---:|
| step_cov | 0.5945 | 0.5558 | -0.0387 |
| step_recall | 0.9633 | 0.9633 | +0.0000 |
| gold_mean | 1.2103 | 1.1495 | -0.0608 |
| multi_at_k | 0.3831 | 0.3795 | -0.3600pp |

### step partition (two-step rows are the pre-registered check)

| partition | queries | domains | A (baseline) | B (candidate) | delta | 95% CI (domain-clustered) |
|---|---:|---:|---:|---:|---:|---|
| 1-step | 75 | 12 | 0.3896 | 0.3765 | -0.0131 | [-0.0410, +0.0183] |
| 2-step | 151 | 13 | 0.4058 | 0.3910 | -0.0147 | [-0.0362, +0.0065] |
| 3-step+ | 258 | 13 | 0.3511 | 0.3336 | -0.0175 | [-0.0456, +0.0100] |
| no-steps | 35 | 10 | 0.1705 | 0.1677 | -0.0028 | [-0.0583, +0.0148] |

### verdict against RESEARCH_PLAN_NEXT 4b

* main criterion (`+0.005`, CI excluding 0): **flat**
* mechanism criterion (step_cov and multi_at_10 move): **not moved**
* per the pre-registration a flat or negative main criterion is reported as *not established*, whatever the mechanism table shows

