## ens3 vs sft_e2bc

| | dev domain-macro nDCG@10 |
|---|---:|
| sft_e2bc (baseline) | 0.3581 |
| ens3 | 0.3567 |
| delta | -0.0014 |

paired bootstrap (`ens3 vs frozen`): delta -0.0014, query CI [-0.0033,+0.0000], domain CI [-0.0032,+0.0001], 19 better / 19 worse

multi-evidence JSON missing for one arm (`scripts/dev_multievidence.py`)

### step partition (two-step rows are the pre-registered check)

| partition | queries | domains | A (baseline) | B (candidate) | delta | 95% CI (domain-clustered) |
|---|---:|---:|---:|---:|---:|---|
| 1-step | 75 | 12 | 0.3896 | 0.3894 | -0.0002 | [-0.0004, +0.0000] |
| 2-step | 151 | 13 | 0.4058 | 0.4030 | -0.0027 | [-0.0071, +0.0009] |
| 3-step+ | 258 | 13 | 0.3511 | 0.3489 | -0.0022 | [-0.0060, +0.0003] |
| no-steps | 35 | 10 | 0.1705 | 0.1705 | +0.0000 | [+0.0000, +0.0000] |

### verdict against RESEARCH_PLAN_NEXT 4b

* main criterion (`+0.005`, CI excluding 0): **flat**
* mechanism criterion (step_cov and multi_at_10 move): **unknown**
* per the pre-registration a flat or negative main criterion is reported as *not established*, whatever the mechanism table shows

