# Checkpoint identifiers (md5 of `best/model.safetensors`)

Every dev number in `results/` comes from one of these directories. `best/` is a
copy of the epoch its run selected; the md5 below identifies the exact weights.

| tag | path | epoch selected | md5 |
|---|---|---|---|
| v2m3_pool8_e2 | `results/ce_ft/v2m3_pool8_e2/best/model.safetensors` | epoch1 | `ff2b4ab8ebe61d0ce787310a91cdf890` (1082 MB) |
| lx_possum | `results/ce_ft/lx_possum/best/model.safetensors` | epoch2 | `64da23087e1022c688dd0dffd04c34ba` (1082 MB) |
| cx_qv | `results/ce_ft/cx_qv/best/model.safetensors` | epoch1 | `2b9df379e10e31c79a10ffa5997f24f6` (1082 MB) |
| cx_bf | `results/ce_ft/cx_bf/best/model.safetensors` | epoch2 | `1afa779cb5467ea04b3fba2bff3fc07e` (1082 MB) |
| sx_l03 | `results/ce_ft/sx_l03/best/model.safetensors` | epoch2 | `64396bd2650983dfc3c430ca76d68c17` (1082 MB) |
| sx_l10 | `results/ce_ft/sx_l10/best/model.safetensors` | epoch2 | `a7801545bd7033cc040b6da6d951abbe` (1082 MB) |

## Where the weights live

The six checkpoints above are 1,082 MB each and are **not** published: they are
not in the repository and no download URL is given. The md5 identifies exactly
which weights produced each dev number, so a reader who has them can verify they
have the right ones, but md5 plus a training command is not a substitute for the
frozen weights themselves.

What *is* published is enough to re-derive every saved ranking without the
weights: the dense candidate runs (`runs/dense_mxbai_*_{dev,train}.trec`), the
cross-encoder top-100 rankings (`runs/rerank_*_dev.trec`), the fused submissions
(`runs/fused_*`, recomputable from the two stages), and the per-query scores
(`results/score_sub_*_dev.json`). Training from the published groups reproduces
the recipe within the 0.347-0.358 retraining band; it does not reproduce a
particular checkpoint bit-for-bit, which is why the md5 table exists.

If the weights are needed for review, publishing one frozen checkpoint
(`v2m3_pool8_e2`) is the cheapest step: it is the system every headline number
in the paper refers to.
