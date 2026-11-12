# Which step-coverage number comes from which list (2026-11-13)

Three different quantities were confused in one draft paragraph (caught by the
second external audit). They are defined here once so the paper, the README and
the JSONs agree.

Notation: a *step* is one annotated evidence requirement of a query; the
denominator of every coverage number below is **all annotated steps with at
least one positive document**, not "steps reachable in the pool".

| quantity | list it measures | files | value |
|---|---|---|---|
| steps retrievable in the dense top-1000 | `runs/dense_mxbai_*_dev.trec` (1000 rows/query) | `MULTIEVIDENCE_*.json` | 96.3% |
| steps retrievable in the reranked top-100 | `runs/rerank_*_dev.trec` (100 rows/query) | `MULTIEVIDENCE_*_depth100.json` | 80.7% |
| step coverage of the fused top ten | `runs/sub_*_dev.trec` (fused, 1000 rows) | `MULTIEVIDENCE_*.json` | 59.5% (frozen) |
| step coverage of the cross-encoder's own top ten | `runs/rerank_*_dev.trec` | `MULTIEVIDENCE_*_depth100.json` | 51.4% (frozen) |
| step coverage of the fused list truncated to 100 rows | `runs/sub100_*` | `MULTIEVIDENCE_sft_e2bc_sub100.json` | 59.45% |

The last row is the control: truncating the fused list from 1000 to 100 rows
leaves the top ten unchanged, so coverage stays 59.45% while reachability drops
from 96.3% to 80.7%. Depth therefore moves recall only; the 51.4%-against-59.5%
gap is a different *ordering* (cross-encoder alone versus dense+CE fusion), not a
different depth.

The suffix `_depth100` on the `MULTIEVIDENCE_*` files is historical: those files
measure the cross-encoder-only ordering, which happens to be 100 rows deep.
