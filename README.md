# RETECO Track 1a — Temporal Retrieval, SemEval-2027 Task 1

Research artifacts for **Track 1a (Temporal Retrieval)** of the
[RETECO](https://datascienceuibk.github.io/RETECO/) shared task:
given a natural-language query and a domain corpus, return a ranked list of
documents that are relevant **and** consistent with the temporal conditions
implied by the query.

**摘要（中文）**：本仓库是 RETECO Track-1a 的全部实验产物：官方 BM25 基线复现、
bge-base 稠密检索、bge-reranker-v2-m3 交叉编码器精排、RRF 两视图融合、跨 13 个领域
的 dev 评测，以及时间表示（时间特征、语义时间一致性、查询分解、HyDE）的一系列负结果。
所有数字都由 TREC run 文件 + 官方 qrels 重算得到，可复现。

## Data

Official de-duplicated v1.1 release (`RETECO-SemEval2027`), Track-1a split:

* 13 domains, **1,167,159 documents** in total (per-domain indices; a single
  query searches only its own domain corpus);
* 519 dev queries / 1,211 train queries; binary qrels (all relevance = 1);
* corpora, prebuilt indexes and model weights are **not** included in this
  repository (size + licensing). See `data/` layout below.

## System

```
query ──► bge-base-en-v1.5 (full query) ──► dense top-1000
                                              │
                        bge-reranker-v2-m3 (title query, 250-token windows)
                                              │
                                     RRF (c=60, w_dense : w_CE = 1:4)
                                              ▼
                                        final ranking
```

* Dense retriever: `BAAI/bge-base-en-v1.5`, CLS pooling, cosine, max length 512,
  query instruction `Represent this sentence for searching relevant passages: `;
* Cross-encoder: `BAAI/bge-reranker-v2-m3`, document windows of 250 tokens
  (stride 250, max pooling), query reduced to the title part (text before the
  first `<p>`), capped at 256 tokens;
* Fusion: reciprocal rank fusion, `c = 60`.

## Results (dev nDCG@10)

| domain | n | dense | CE | RRF 1:4 |
|---|---|---|---|---|
| bitcoin | 30 | 0.2171 | 0.2121 | **0.2349** |
| economics | 25 | 0.1891 | 0.1664 | **0.1995** |
| iota | 3 | 0.4712 | — | — |
| cardano | 15 | 0.1496 | 0.1921 | **0.2232** |
| genealogy | 35 | 0.2110 | 0.2473 | **0.2762** |
| history | 240 | 0.2772 | 0.3153 | 0.3236 |
| hsm | 45 | 0.2926 | 0.3976 | 0.3974 |
| law | 11 | **0.3013** | 0.2152 | 0.2118 |
| monero | 19 | 0.1328 | 0.1638 | 0.1710 |
| politics | 45 | 0.3074 | **0.4051** | 0.4047 |
| quant | 10 | 0.0564 | 0.0727 | 0.0825 |
| travel | 30 | 0.2466 | 0.2416 | 0.2596 |
| workplace | 11 | **0.3795** | 0.2547 | 0.2864 |

Pooled over the 12 domains with a cross-encoder (516 query–domain pairs),
query-level paired bootstrap:

| method | Δ vs dense | 95% CI | P(Δ>0) |
|---|---|---|---|
| CE only | +0.0344 | [+0.0143, +0.0545] | 0.9998 |
| RRF 2:1 | +0.0305 | [+0.0212, +0.0401] | 1.000 |
| RRF 1:2 | +0.0423 | [+0.0265, +0.0582] | 1.000 |
| **RRF 1:4** | **+0.0461** | [+0.0288, +0.0635] | 1.000 |

Macro nDCG@10: dense **0.2300**, CE 0.2403, two-view **0.2559**.
Leave-one-domain-out weight selection picks 1:4 in 10/12 folds and yields a
mean held-out gain of **+0.0150** — i.e. the honest cross-domain effect size is
smaller than the pooled number.

## What we learned (short version)

1. **The official BM25 baseline is not Lucene BM25.** The starter kit applies
   `gensim`'s `LuceneBM25Model` to the query as well, so the dot product carries
   idf twice (plus a query-length normaliser). This explains the published
   numbers exactly (Bitcoin dev 0.0370, IOTA dev 0.3309, macro 0.1147), while a
   real Anserini index on the same corpus gives 0.0300 for Bitcoin. See
   `results/BASELINE_FORENSICS.md`.
2. **Dense is worth ~2.4× the published BM25 macro** (0.2300 vs ≈0.0967 on the
   same 12 domains).
3. **CE-dominant fusion is a consistent but small win** (+0.026 macro, 10/12
   domains). It fails on `law` and `workplace` (CE alone is worse than dense on
   those), so a domain gate is needed before using it blindly.
4. **Text-only temporal signals do not work on this benchmark.** Document
   timestamps do not exist; only 12–25 % of gold documents contain a year.
   Content-time features, a semantic temporal-consistency score, query
   decomposition (oracle and LLM) and HyDE were all neutral or negative. See
   `results/TIME_ABLATION.md`, `results/TEMPORAL_SEMANTIC_PROBE.md`,
   `results/TEMPORAL_DECOMPOSITION.md`.
5. **Listwise LLM rerankers echo their input order** on these long forum
   queries (reversed-input control reproduces `[1] > [2] > ...`), so only
   pairwise prompting carries signal — and it is unstable across train/dev.

## Repository layout

```
docs/RESEARCH_LOG.md          full research log (questions, rules, phase narratives)
docs/PHASE0.md                environment, reproduction commands, caveats
results/*.md                  per-phase reports (11 documents)
results/*.json                every run's score, sweep, meta and summary
runs/*.trec                   TREC run files (dense / CE / fused / BM25)
scripts/*.py, scripts/*.sh    all experiment code
logs/*.log                    run logs
```

`data/`, `models/`, `emb/`, `indexes/` are intentionally absent.

All scripts resolve their project root from the `RETEco_ROOT` environment
variable and fall back to the repository root, so a checkout works as long as
the data/models trees are placed next to it:

```bash
export RETEco_ROOT=/path/to/RETECO        # optional; defaults to the repo root
export RETEco_MODELS=$RETEco_ROOT/models  # optional; local weights directory
```

## Reproduce

Environment: Python 3.12, Java 21 (Temurin), `pyserini` 2.4.0, `pyjnius` 1.9.0,
`torch` 2.7.0+cu126, `transformers` 5.22.2, `bge-base-en-v1.5`.

```bash
# 1. data (official release), one domain at a time
#    data/track1_tempo/<domain>/{documents.jsonl,examples_dev.jsonl,qrels_dev.txt,guidance_dev.jsonl}

# 2. dense retrieval (encodes the corpus once, caches the matrix)
python scripts/run_dense_bge.py --index-data data/track1_tempo/<domain> --split dev \
    --model models/bge-base-en-v1.5 --work emb \
    --out runs/dense_bge_<domain>_dev.trec \
    --meta runs/dense_bge_<domain>_dev.meta.json

# 3. cross-encoder rerank of the top-100
python scripts/run_chunk_rerank.py --domain <domain> --split dev \
    --run runs/dense_bge_<domain>_dev.trec --k 100 --q-cap 256 --q-mode title \
    --win 250 --stride 250 --pool max --model models/bge-reranker-v2-m3 \
    --out runs/rerank_v2m3_title_<domain>_dev.trec \
    --meta runs/rerank_v2m3_title_<domain>_dev.meta.json

# 4. fuse + score
python scripts/fuse_rrf.py --run-a runs/dense_bge_<domain>_dev.trec \
    --run-b runs/rerank_v2m3_title_<domain>_dev.trec \
    --out runs/twoview_<domain>_dev.trec --wa 1 --wb 4 --c 60
PYTHONUTF8=1 python scripts/score_run.py --run runs/twoview_<domain>_dev.trec \
    --data data/track1_tempo/<domain> --split dev --json results/score_twoview_<domain>_dev.json

# 5. cross-domain analysis (bootstrap, fusion weights, LOOCV)
PYTHONUTF8=1 python scripts/analyze_phase7.py --fusions "2,1;1,1;1,2;1,3;1,4"
```

On Chinese Windows, `PYTHONUTF8=1` is required for the official `scorer.py`
(it decodes the `steps_*.jsonl` files as GBK otherwise).

## Caveats

* Everything here is **dev-set** measurement. Fusion weights and reranking
  configurations were selected on dev; LOOCV mitigates but does not remove this.
* `iota` has 3 dev queries (and 7 train queries); its numbers are not
  informative.
* The hidden test set is not used anywhere in this repository.
* No document-ID heuristics are used (the organizers warned that ID patterns
  correlate with positive labels in train/dev).
* The two-view fusion is positive in 10/12 domains but the **domain-clustered**
  interval crosses zero ([−0.0097, +0.0581]), and LOOCV's honest gain is +0.0150.
  Read it as "usually positive", not "safely transfers to new domains".
* `law` and `workplace` degrade under CE: the cross-encoder is OOD on their
  long/legal and procedural texts. A domain/confidence gate is future work.

## Known gaps (in progress)

1. **No comparison to strong published retrievers.** The TEMPO benchmark paper
   reports E5 ≈ 0.304 and DiVeR ≈ 0.320 nDCG@10, but on a different corpus
   setting; these numbers are not directly comparable to RETECO v1.1 dev. A
   same-version E5 (and any feasible of DiVeR/ReasonIR) baseline is being added.
2. **Temporal reasoning is not demonstrated.** The negative results above probe
   *content-time* signals; they do not test whether a reranker can separate
   "same topic, different time" documents, because no such hard negatives are
   constructed yet. That experiment is planned.
3. **History is squashed.** The published repository is a single curated commit;
   it does not preserve development history. All raw artifacts (runs, scores,
   logs) are included so the numbers remain checkable.

## Data & license

Data and baselines are from the RETECO organizers
(`DataScience-UIBK/RETECO-SemEval2027`, `castorini/prebuilt-indexes-reteco`).
`scripts/ir_metrics.py` is a vendored copy of the official starter-kit scorer
(see its header); the rest of the starter kit is **not** vendored here — see
<https://github.com/DataScienceUIBK/RETECO>.
