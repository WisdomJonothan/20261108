# Weight sensitivity of the SFT gain (dev, domain-macro nDCG@10)

Fusion: `s = z(dense) + w * z(CE)` per query, same candidate set. The
published weight is w=1; it was fitted on train with the ZERO-SHOT CE only.
Diagnostic sweep, nothing selected.

| w (CE) | zero-shot | SFT | delta | 95% domain-clustered CI | domains up |
|---:|---:|---:|---:|---|---:|
| 0.5 | 0.3208 | 0.3297 | +0.0089 | [+0.0009, +0.0170] | 9/13 |
| 0.7 | 0.3294 | 0.3472 | +0.0178 | [+0.0102, +0.0261] | 13/13 |
| 0.85 | 0.3261 | 0.3518 | +0.0258 | [+0.0158, +0.0351] | 11/13 |
| 1 | 0.3249 | 0.3581 | +0.0332 | [+0.0195, +0.0482] | 11/13 |
| 1.15 | 0.3271 | 0.3552 | +0.0280 | [+0.0168, +0.0393] | 11/13 |
| 1.3 | 0.3278 | 0.3559 | +0.0281 | [+0.0174, +0.0387] | 11/13 |
| 1.5 | 0.3253 | 0.3592 | +0.0339 | [+0.0230, +0.0451] | 13/13 |
| 1.8 | 0.3192 | 0.3538 | +0.0346 | [+0.0194, +0.0535] | 13/13 |
| 2 | 0.3193 | 0.3481 | +0.0287 | [+0.0118, +0.0498] | 11/13 |

Mean Spearman (dense vs SFT CE) over the top-100: 0.279.

## Per domain at the published weight (w=1)

| domain | zero-shot | SFT | delta |
|---|---:|---:|---:|
| bitcoin | 0.2166 | 0.2143 | -0.0023 |
| cardano | 0.1710 | 0.2258 | +0.0547 |
| economics | 0.2122 | 0.2512 | +0.0390 |
| genealogy | 0.3760 | 0.4060 | +0.0299 |
| history | 0.3464 | 0.3918 | +0.0454 |
| hsm | 0.3938 | 0.4651 | +0.0713 |
| iota | 0.8254 | 0.8383 | +0.0129 |
| law | 0.3596 | 0.4357 | +0.0760 |
| monero | 0.1993 | 0.2069 | +0.0076 |
| politics | 0.4086 | 0.4290 | +0.0204 |
| quant | 0.1547 | 0.1538 | -0.0009 |
| travel | 0.2352 | 0.2494 | +0.0142 |
| workplace | 0.3249 | 0.3881 | +0.0631 |
