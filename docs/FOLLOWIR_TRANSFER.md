# FollowIR transfer: feasibility on this host (2026-11-20)

Status: **not executed**. This note records why, with the probes that were run, so
the claim is not silently dropped from the paper.

## What the official protocol needs

FollowIR evaluates instruction-following rerankers on MSMARCO-v1/v2 and TREC
NeuCLIR collections: for each topic it needs the *original* and *modified*
instruction, the qrels for both, the provided BM25 candidate runs, **and the
document texts** (a reranker has to read the passages). The official evaluation
split is hosted on Hugging Face (`jhu-clsp/FollowIR`); the corpora come from the
MSMARCO/BEIR distributions.

## Probes run from the RETECO server (10.13.71.195)

| endpoint | result |
|---|---|
| `https://huggingface.co/datasets/jhu-clsp/FollowIR` | `000` (no route) |
| `https://modelscope.cn/api/v1/datasets/jhu-clsp/FollowIR` | `000` (no route) |
| `https://github.com` / `https://api.github.com` | `000` (no route) |
| `https://raw.githubusercontent.com/WisdomJonothan/20261108/main/README.md` | `200`, 32,233 B (our own repo works) |
| `https://raw.githubusercontent.com/jhu-clsp/FollowIR/main/README.md` | `404` (reachable path, wrong repo/branch; the FollowIR data tree is not there) |

The RETECO box sits behind a whitelist: PyPI, several university/corporate
mirrors, and `raw.githubusercontent.com` answer; Hugging Face, ModelScope and
`github.com` do not. ModelScope mirrors `jhu-clsp/FollowIR-train` and the
`mFollowIR-*` variants, but the official FollowIR evaluation split is not among
them (the ModelScope API answers "dataset does not exist" for
`jhu-clsp/FollowIR`), and the MSMARCO corpora needed for passage text are not
mirrored on any reachable host either.

A faithful FollowIR run would therefore require moving tens of GB of MSMARCO
passages through the Windows workstation that relays the RETECO models (the same
machine reaches ModelScope at ~3.5 MB/s), and even then the evaluation split
itself is not obtainable from a reachable source. That is outside this study's
budget.

## What is reported instead

1. **Cross-domain transfer inside RETECO** — the per-domain nDCG@10 table in
   `results/replacement_v2/SUMMARY_DEV.md` shows the effect on each of the 13
   domains separately (Direct improves 11 of 13, largest on monero +0.0166,
   travel +0.0202, law +0.0176; bitcoin and workplace move down).
2. **Second starting system** — the same frozen protocol applied to the dense
   ranking (`results/replacement_v2/exp_dense_start_dev.json`), i.e. a system
   that is 0.0638 nDCG@10 weaker and has no cross-encoder in the pool.
3. **Budget curve** — 0/1/2/3 candidates (`exp_budget_cap{1,2}.json`), which is
   the compute side of the same transfer question.

## Overlap with the training queries

RETECO's replacement candidates never read qrels, and the study trains nothing on
dev, so there is no RETECO-train/dev overlap to control for. A FollowIR transfer
would have raised the separate question of MSMARCO-v1 topics appearing in both
FollowIR and its training data; that check remains open because the data could
not be obtained.
