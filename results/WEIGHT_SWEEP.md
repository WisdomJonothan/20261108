# Weight sensitivity of the SFT gain (dev, domain-macro nDCG@10)

Fusion: `s = z(dense) + w * z(CE)` per query, same candidate set. The
published weight is w=1; it was fitted on train with the ZERO-SHOT CE only.
Diagnostic sweep, nothing selected.

| w (CE) | zero-shot | SFT | delta | 95% domain-clustered CI | domains up |
|---:|---:|---:|---:|---|---:|
| 0.5 | 0.3208 | 0.3211 | +0.0004 | [-0.0031, +0.0042] | 5/13 |
| 0.7 | 0.3294 | 0.3287 | -0.0007 | [-0.0081, +0.0053] | 7/13 |
| 0.85 | 0.3261 | 0.3308 | +0.0047 | [+0.0001, +0.0098] | 8/13 |
| 1 | 0.3249 | 0.3336 | +0.0087 | [+0.0038, +0.0142] | 10/13 |
| 1.15 | 0.3271 | 0.3319 | +0.0048 | [+0.0008, +0.0089] | 9/13 |
| 1.3 | 0.3278 | 0.3334 | +0.0056 | [+0.0004, +0.0114] | 10/13 |
| 1.5 | 0.3253 | 0.3298 | +0.0045 | [-0.0018, +0.0108] | 8/13 |
| 1.8 | 0.3192 | 0.3341 | +0.0149 | [+0.0055, +0.0293] | 12/13 |
| 2 | 0.3193 | 0.3274 | +0.0080 | [+0.0041, +0.0122] | 11/13 |

Mean Spearman (dense vs SFT CE) over the top-100: 0.328.

## Per domain at the published weight (w=1)

| domain | zero-shot | SFT | delta |
|---|---:|---:|---:|
| bitcoin | 0.2166 | 0.2085 | -0.0080 |
| cardano | 0.1710 | 0.2047 | +0.0336 |
| economics | 0.2122 | 0.2157 | +0.0035 |
| genealogy | 0.3760 | 0.3838 | +0.0077 |
| history | 0.3464 | 0.3547 | +0.0083 |
| hsm | 0.3938 | 0.4120 | +0.0182 |
| iota | 0.8254 | 0.8344 | +0.0090 |
| law | 0.3596 | 0.3691 | +0.0094 |
| monero | 0.1993 | 0.2048 | +0.0055 |
| politics | 0.4086 | 0.4074 | -0.0011 |
| quant | 0.1547 | 0.1547 | +0.0000 |
| travel | 0.2352 | 0.2475 | +0.0123 |
| workplace | 0.3249 | 0.3391 | +0.0141 |
