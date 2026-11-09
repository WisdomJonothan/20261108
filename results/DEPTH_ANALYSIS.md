# Candidate depth: how much headroom is left outside the top-100?

All numbers are dev, 13 domains, domain-macro (official aggregation), recomputed
from `runs/dense_mxbai_<domain>_dev.trec` (1,000 documents per query) and the
official `ir_metrics.ndcg_at_k`. Script: `scripts/depth_probe.py`; the table can
be reproduced in seconds with the snippet at the bottom.

## Recall inside the dense pool

| depth k | Recall@k | oracle nDCG@10 of the pool | gap to dense@10 |
|---:|---:|---:|---:|
| 10 | 0.3852 | 0.3852 | 0.0909 |
| 20 | 0.4844 | 0.4844 | 0.1901 |
| 50 | 0.6086 | 0.6086 | 0.3143 |
| **100** | **0.6995** | **0.7446** | **0.4503** |
| 200 | 0.7783 | 0.8150 | 0.5208 |
| 300 | 0.8183 | 0.8513 | 0.5570 |
| 500 | 0.8697 | 0.8939 | 0.5996 |
| 1000 | 0.9247 | 0.9416 | 0.6473 |

dense@10 is 0.2943 at every depth (the dense list only ever contributes its top
10). "oracle" orders the whole pool ideally; at k=100 it reproduces the 0.7446 of
`results/CAPTURE_CURVE.md`, so the two analyses agree.

**Reading.** The frozen system captures 8.7 % of the top-100 headroom. Going from
100 to 300 candidates adds 0.1067 of oracle nDCG@10 (0.7446 → 0.8513) and lifts
recall from 0.6995 to 0.8183; 22.5 % of dev gold documents sit between dense rank
100 and 1000. If the CE kept its current capture rate, a 300-deep pool would score
0.2943 + 0.087 × 0.557 = **0.3428** (+0.009 over the frozen 0.3336), and a
500-deep pool +0.052. The open question is exactly whether the capture rate
survives at depth: candidates at rank 200–300 are much harder negatives than the
top-100 the checkpoint was trained on, so the CE may score them near the floor.

## Probe (the cheap way to find out)

`scripts/depth_probe.py --select` picks, per domain, up to 6 dev queries with
0 < Recall@100 < 1 and Recall@500 > Recall@100 (gold still hidden at depth),
hardest first — 62 queries in total (bitcoin 6, cardano 5, economics 6, genealogy 6,
history 6, hsm 6, iota 1, law 2, monero 5, politics 6, quant 5, travel 6,
workplace 2). `scripts/run_depth_probe.sh` re-ranks the 200-deep pool for exactly
those queries with the SFT checkpoint (~25–35 min GPU, versus 3–5 h for a full
dev rerank at k=300), then `--score` compares, on the same queries, fused z 1:1
nDCG@10 at pool 100 vs pool 200 with a query bootstrap.

Decision rule, fixed before looking at the result: if the 200-deep pool does not
beat the 100-deep pool by at least +0.01 on these 62 queries with a bootstrap
interval excluding zero, depth is not worth the compute for the January run and
the story stays "ranking, not recall". If it does, the full dev rerank at k=300
becomes the next submission candidate.

## Note on the 27.7 % zero-score queries

`results/DEV_ERROR_ANALYSIS.md` splits the 144 zero-nDCG queries into 92 with gold
inside the top-100 (ordering damage) and 52 with Recall@100 = 0 (retrieval
failure). The depth table says most of those 52 are rescueable in principle:
Recall@1000 = 0.9247, so their gold documents exist in the dense list further
down. Depth is therefore the only lever that can touch the retrieval-failure half
of the error budget; fusion weights and CE training can only touch the other half.

## Result (2026-11-09 16:18): depth does NOT help

`scripts/run_depth_probe.sh` (e1 checkpoint, 200-deep pool, 62 queries):

| domain | n | pool 100 | pool 200 | Δ |
|---|---:|---:|---:|---:|
| bitcoin | 6 | 0.1224 | 0.1206 | −0.0018 |
| cardano | 5 | 0.1028 | 0.1043 | +0.0015 |
| economics | 6 | 0.2596 | 0.2596 | 0.0000 |
| genealogy | 6 | 0.2585 | 0.2639 | +0.0055 |
| history | 6 | 0.0714 | 0.0714 | 0.0000 |
| hsm | 6 | 0.3249 | 0.3249 | 0.0000 |
| iota | 1 | 0.8319 | 0.8319 | 0.0000 |
| law | 2 | 0.1265 | 0.1265 | 0.0000 |
| monero | 5 | 0.1253 | 0.1253 | 0.0000 |
| politics | 6 | 0.1841 | 0.1841 | 0.0000 |
| quant | 5 | 0.2731 | 0.2731 | 0.0000 |
| travel | 6 | 0.1717 | 0.1771 | +0.0054 |
| workplace | 2 | 0.1092 | 0.1092 | 0.0000 |
| **overall** | **62** | — | — | **+0.0010, CI [−0.0009, +0.0036], 3 up / 2 down** |

The delta is zero in 9 of 13 domains: the cross-encoder scores the rank-100–200
candidates low enough that they never enter the top 10, so a deeper pool changes
nothing. Deeper candidates would only pay off after fine-tuning **on** them
(training currently uses top-100 negatives). Decision per the pre-registered
rule: do not spend the 3–5 GPU-hours of a k=300 dev rerank; the paper's claim
"the headroom is ranking, not recall" survives, with the sharper statement that
the frozen CE cannot exploit new candidates it was not trained on.
