| partition | queries | domains | A (baseline) | B (candidate) | delta | 95% CI (domain-clustered) |
|---|---:|---:|---:|---:|---:|---|
| 1-step | 75 | 12 | 0.3185 | 0.3896 | +0.0711 | [+0.0244, +0.1056] |
| 2-step | 151 | 13 | 0.3857 | 0.4058 | +0.0200 | [-0.0033, +0.0463] |
| 3-step+ | 258 | 13 | 0.3196 | 0.3511 | +0.0315 | [+0.0084, +0.0571] |
| no-steps | 35 | 10 | 0.1579 | 0.1705 | +0.0126 | [+0.0030, +0.0524] |

Overall delta (all dev queries, domain-macro): +0.0332

- 1-step: 75 queries (14.5%), delta +0.0711
- 2-step: 151 queries (29.1%), delta +0.0200
- 3-step+: 258 queries (49.7%), delta +0.0315
- no-steps: 35 queries (6.7%), delta +0.0126
