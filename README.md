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
query ──► mixedbread-ai/mxbai-embed-large-v1 (full query) ──► dense top-100
                                              │
                        bge-reranker-v2-m3 (title query, 250-token windows)
                                              │
                              RRF (c = 60, w_dense : w_CE = 1 : 1)
                                              ▼
                                        final ranking
```

* Dense retriever: `mixedbread-ai/mxbai-embed-large-v1`, mean pooling, cosine,
  max length 512, query instruction
  `Represent this sentence for searching relevant passages: ` (fp16);
* Cross-encoder: `BAAI/bge-reranker-v2-m3`, document windows of **250 tokens,
  stride 192** (max pooling over windows), query reduced to the title part
  (text before the first `<p>`), capped at 256 tokens. The earlier release
  described stride 250; every run's `meta.json` says 192 and that is what the
  committed runs used;
* Fusion: reciprocal rank fusion, `c = 60`, `w_dense : w_CE = 1 : 1`
  (`scripts/fuse_rrf.py --wa 1 --wb 1 --c 60`; the script defaults to 2:1, so
  pass the weights explicitly).

**Reporting convention (corrected 2026-11-09).** The official Track-1a
aggregation is **domain-macro**: the Evaluation Plan says nDCG@10 is
"computed for each query, averaged over the queries of each domain, and then
averaged over the 13 domains, so every domain counts equally", and missing
queries score 0. `iota` (3 dev queries) therefore weighs as much as `history`
(240). `topic-macro` (519-query mean) is reported as a diagnostic only. Both
are computed by `scripts/aggregate_scores.py`; every table states which it uses.

| frozen system (V1), dev | domain-macro (official) | topic-macro |
|---|---:|---:|
| **dense + zero-shot CE, RRF 1:1** | **0.3166** | 0.3199 |
| domain count | 13 | 519 |

Official Track-1a dev BM25 baseline is **0.1147**, i.e. this stack is ≈2.8× the
official lexical baseline. TEMPO's best reported system (32.0) is on a different
1,730-query split and is not directly comparable.

**Phase 8 update (recommended configuration).** Swapping only the first stage to
`mixedbread-ai/mxbai-embed-large-v1` and fusing `w_dense : w_CE = 1:1` lifts the
dev macro over the **shared 11 domains** from 0.2556 to **0.2843**; over all
13 domains the fused macro is **0.3166** (dense 0.2943). The two 12-domain
macros quoted in the first release are not directly comparable — they swap
`travel` for `iota`, and `iota` has only 3 dev queries (see Phase 8 below).

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

> **Committed artifacts.** Dev TREC runs for every system are in `runs/`.
> The 7-domain **train** runs (83 MB) are not committed; regenerate them with
> `bash scripts/phase8_train_mxbai.sh` (dense + CE) and
> `bash scripts/phase8_train_features.sh` (feature rows) before running
> `scripts/temporal_gate_fit.py`.

## Phase 8 — stronger first stage (dev nDCG@10)

Motivation: the first release had no comparison against retrievers stronger than
bge-base. TEMPO (arXiv:2601.09523v1) reports E5 30.4 / DiVeR-4B 32.0, but those
are 7B/4B models on its own 1,730-query v1.0 corpus, so they cannot be quoted
side by side here (only BM25 0.1147 is directly comparable). We instead ran
zero-shot 8 GB-friendly encoders on the identical dev split.

| domain | bge-base | e5-base | mxbai | mxbai+CE 1:1 |
|---|---:|---:|---:|---:|
| bitcoin | 0.2171 | 0.1224 | 0.1820 | 0.2044 |
| economics | 0.1891 | 0.0525 | 0.1890 | 0.2099 |
| politics | 0.3074 | 0.2264 | **0.3599** | 0.4023 |
| genealogy | 0.2110 | 0.1436 | **0.3298** | 0.3372 |
| hsm | 0.2926 | 0.1413 | **0.3125** | 0.3840 |
| history | 0.2772 | 0.2413 | **0.3051** | 0.3441 |
| law | 0.3013 | 0.0633 | **0.3689** | 0.3574 |
| workplace | **0.3795** | 0.1046 | 0.3362 | 0.3244 |
| cardano | 0.1496 | 0.1312 | **0.1832** | 0.1632 |
| monero | 0.1328 | 0.1139 | **0.1826** | 0.2281 |
| quant | 0.0564 | 0.0000 | **0.1954** | 0.1631 |
| travel | 0.2466 | 0.1653 | 0.2321 | 0.2205 |
| iota (3 q) | 0.4712 | 0.4183 | **0.6485** | 0.7771 |
| **macro (13)** | 0.2486 | 0.1480 | **0.2942** | **0.3166** |

* `intfloat/e5-base-v2`: macro **0.148** over all 13 domains — mean pooling over
  these long HTML documents is simply weak (the scoring path was verified
  against bge, which reproduces 0.3013 on `law`).
* `mxbai-embed-large-v1`: macro **0.2942** over all 13 domains; wins 10/13
  domains, loses `bitcoin` (−0.035), `workplace` (−0.043) and `travel` (−0.015).
* `Alibaba-NLP/gte-large-en-v1.5` could not be run: its custom RoPE/pack_qkv code
  raises `AttributeError: 'NewModel' object has no attribute
  'get_extended_attention_mask'` under transformers 5.22.

Two-view pipeline on top of mxbai (same CE, same RRF):

| fusion | macro | Δ vs dense | 95% CI (query) | domain-clustered CI |
|---|---:|---:|---|---|
| CE only | 0.2680 | +0.0099 | [−0.0111, +0.0307] | — |
| RRF 2:1 | **0.3176** | +0.0271 | [+0.0163, +0.0377] | [+0.0009, +0.0492] |
| RRF 1:1 | 0.3166 | **+0.0307** | [+0.0166, +0.0449] | [+0.0012, +0.0471] |
| RRF 1:2 | 0.3134 | +0.0334 | [+0.0164, +0.0503] | [−0.0162, +0.0541] |
| RRF 1:4 | 0.2981 | +0.0297 | [+0.0103, +0.0486] | [−0.0364, +0.0416] |

Macro over 13 domains; query-level bootstrap over 519 queries. LOOCV honest
gain is only **+0.0087**, because per-domain optima differ (`law`, `quant`
prefer CE; `workplace`, `iota` prefer dense).

**Choosing the weight on train.** With z-normalised score fusion the train-side
(1,211 queries) optimum under the official domain-macro statistic is **1:1**
(train 0.2612 vs 0.2596 for 1:2); applied unchanged to dev it gives 0.3249, i.e.
+0.0083 over the RRF baseline with zero dev-side tuning. (The RRF table above
and its "train optimum 1:2 / 0.2519" line were computed under **topic**-macro,
where the CE wants more weight; that ordering was an artifact of the wrong
statistic and is kept only as a record.) Per-domain weights chosen on each
domain's own train split do **not** transfer (dev 0.2941 ≈ dense), so one global
weight is the robust choice.

**Same 11 domains, both encoders** (`scripts/fair_11dom_compare.py`) — the only
apples-to-apples comparison, 486 queries:

| system | domain macro |
|---|---:|
| bge-base dense | 0.2285 |
| bge-base + CE, RRF 1:4 | 0.2556 |
| mxbai dense | 0.2677 |
| mxbai + CE, RRF 1:1 | **0.2843** |

Encoder upgrade +0.0287 (+12.6 %). Fusion gain on mxbai +0.0166, query CI
[+0.0046, +0.0291], but **domain-clustered [−0.0026, +0.0345] — still crosses
zero**, so the fusion gain is not established beyond dev.

Full tables, caveats and reproduction commands:
`results/PHASE8_STRONGER_BASELINES.md`.

## Temporal observability and ceiling (Phase 8)

Document timestamps do not exist in RETECO; only 12–25 % of gold documents
contain a year. Coverage is highly uneven: politics 84.7 %, economics 64.2 %,
genealogy 56.4 %, hsm 52.2 %, history 37.5 % vs bitcoin 19.1 %, cardano 9.3 %,
quant 14.3 %. Earlier temporal modules were therefore evaluated mostly where
temporal consistency is nearly unobservable.

Ceiling test (no qrels, reorder dense top-100 by query-year vs doc-year
consistency, `scripts/temporal_ceiling.py`): demoting "year-mismatched" documents
*decreases* nDCG@10 in most domains (politics 0.307→0.268, genealogy 0.211→0.184,
hsm 0.293→0.271, history 0.277→0.269, bitcoin 0.217→0.197) and only helps
economics (0.189→0.201). Among year-bearing candidates 53–100 % carry a year
outside the query interval, and many of those are gold: **time mismatch is not
irrelevance** on this benchmark. (An earlier "infeasible" claim about hard
negatives rested on a parse bug that read the TREC rank column instead of the
score; after the fix mining yields **1,857** topic-hard/time-hard pairs over
12 domains.) A pre-registered temporal gate — logistic head fitted on train,
λ chosen inside train, dev read once — reaches ΔnDCG@10 **+0.0039**
(CI [−0.0023, +0.0103]), far below the +0.02 adoption bar: the temporal
statement is discriminative (s_temp AUC 0.61–0.88) but does not convert into
ranking gains. See `results/TEMPORAL_CEILING.md`,
`results/TEMPORAL_OBSERVABILITY.md` and `results/TEMPORAL_GATE.md`.

## Phase 9 — supervised re-ranking, train split, and submission freeze

**Data gap fixed.** The project had been built on a partial download: five
domains (`cardano, law, quant, travel, workplace`) had no train files, so
"only 8 of 13 domains have train" was wrong. The official release is
**1,211 train / 519 dev** over all 13 domains (70/30 per domain); after
`scripts/fetch_missing_train.py` the local counts match exactly. The corpus is
v1.1 (13-domain total 1,167,159 documents, identical to the organizers' figure).

**Diagnosis.** Inside the frozen top-100 pool the oracle nDCG@10 is **0.745**
while the system scores 0.320 — about 5 % of the ranking headroom. The zero-shot
cross-encoder's within-query AUC (0.801) is no better than the dense retriever's
(0.806), and ten post-hoc interventions failed. The binding constraint is the
scorer, not recall (a second dense retriever lifts Recall@100 0.699 → 0.737 and
oracle@10 0.745 → 0.778, but a 5 %-capture ranker converts that into ≈+0.002).

**Fusion form.** RRF discards score magnitude and taxes the stronger component.
Fusing on within-query z-normalised scores instead, over the same candidates:

| fusion (same candidates) | domain-macro (official) | topic-macro | paired bootstrap (query, domain-macro stat) |
|---|---:|---:|---|
| RRF 1:1 (V1) | 0.3166 | 0.3199 | — |
| score-sum 1:1 | **0.3249** | 0.3260 | +0.0083 [−0.0010,+0.0179]; domain-clustered [−0.0015,+0.0188] |
| score-sum 1:2 | 0.3193 | 0.3302 | — |

The weight is fitted on train under the official statistic, which selects
**1:1** (train domain-macro 0.2612 vs 0.2596 for 1:2). Under topic-macro the CE
wants more weight (1:2), which is why the earlier 1:2 choice was an artifact of
the wrong metric. (`scripts/fit_fusion_weight.py --select-metric`,
`scripts/bootstrap_compare.py --stat domain`.)

**Submission freeze.** The 13 fused dev runs are committed as
`runs/sub_1_1_<domain>_dev.trec`, de-tied (score − 1e-9·rank) so a
score-sorting scorer cannot permute tied top-10 slots, and validated by the
official `format_checker.py` with `--qrels` **and** `--corpus` through
`scripts/validate_submission.py`: 13/13 files, 0 errors. `score_run.py` no longer
silently drops topics missing from a run (the leaderboard counts them as 0).

**Supervised fine-tuning.** Everything above is zero-shot. With 1,211 labelled
train queries the cross-encoder can now be trained on-domain
(`scripts/build_ce_train_groups.py` → `scripts/train_ce.py`: listwise softmax,
all in-candidate positives + 70 % hard negatives, bf16 + gradient checkpointing,
5.5 GB peak on one 8 GB GPU, epoch selection on held-out **train** queries).
A CE-only gain passes through fusion at ≈19 % (RRF 1:1), 26 % (score-sum) or
61 % (RRF 1:4), so the fusion weight must be refit on train.

Result under the official metric, at the train-selected weight 1:1:

| dev (519 topics) | domain-macro | topic-macro |
|---|---:|---:|
| dense + zero-shot CE, z 1:1 | 0.3249 | 0.3260 |
| dense + SFT CE, z 1:1 | **0.3336** | 0.3339 |

SFT vs zero-shot at the fixed weight: **+0.0087**, query-bootstrap CI
[+0.0025,+0.0151], domain-clustered [+0.0037,+0.0142] — the only single-factor
effect whose domain-clustered CI excludes zero. The total over V1 is
**+0.0170** [+0.0055,+0.0290], domain-clustered [+0.0052,+0.0287], but it
bundles that +0.0087 with a fusion-form change worth +0.0083 whose own CI
crosses zero, so the honest single claim is +0.0087 over the strongest
zero-shot baseline.

The pre-registered adoption bar (+0.015 on dev **topic-macro**) was **not** met:
under the registered statistic SFT − V1 = 0.3339 − 0.3199 = +0.0141. The bar is
not restated under the corrected metric; the significance claims above are made
with the official domain-macro bootstrap instead. Frozen as
`runs/sub_sft11_<domain>_dev.trec` (13/13 official-checked, de-tied).
See `results/PHASE9_PLAN.md` and `docs/RESEARCH_LOG.md` §5i.

**The SFT arm inherited a weight tuned on the zero-shot arm.** One global scalar
was fitted once on the 1,211 train queries with the zero-shot CE (1:1), frozen,
and never re-fitted; the SFT arm is therefore the *under-tuned* one, so its
measured gain is conservative rather than inflated. To show that the claim does
not hinge on that choice, `scripts/weight_sweep.py` re-fuses the stored
per-query scores at nine weights for **both** arms on dev (diagnostic only —
nothing is selected on dev):

| w (CE) | zero-shot | SFT | delta | 95% domain-clustered CI | domains up |
|---:|---:|---:|---:|---|---:|
| 0.5 | 0.3208 | 0.3211 | +0.0004 | [−0.0031,+0.0042] | 5/13 |
| 0.7 | 0.3294 | 0.3287 | −0.0007 | [−0.0081,+0.0053] | 7/13 |
| 0.85 | 0.3261 | 0.3308 | +0.0047 | [+0.0001,+0.0098] | 8/13 |
| **1 (frozen)** | 0.3249 | **0.3336** | **+0.0087** | **[+0.0038,+0.0142]** | 10/13 |
| 1.15 | 0.3271 | 0.3319 | +0.0048 | [+0.0008,+0.0089] | 9/13 |
| 1.3 | 0.3278 | 0.3334 | +0.0056 | [+0.0004,+0.0114] | 10/13 |
| 1.5 | 0.3253 | 0.3298 | +0.0045 | [−0.0018,+0.0108] | 8/13 |
| 1.8 | 0.3192 | 0.3341 | +0.0149 | [+0.0055,+0.0293] | 12/13 |
| 2.0 | 0.3193 | 0.3274 | +0.0080 | [+0.0041,+0.0122] | 11/13 |

Both arms move by ~±0.008 across the sweep with no monotone trend, i.e. the
fusion surface is flat to within sampling noise; the SFT arm is ahead at 7 of 9
weights and at every weight ≥ 0.85. We report the frozen-weight number, not a
dev-best one. (`results/WEIGHT_SWEEP.md`, `results/weight_sweep_dev.json`.)

**Disclosure: the SFT validation split is not recoverable.** Training held out
45 train queries (seed 17, 12 % per domain capped at 4) and selected nothing on
them (one epoch, fixed config). Replaying that split from the current code does
not reproduce the logged per-domain `docs_loaded` counts — all 13 domains, seeds
0–60 — so the exact 45 are unknown. They are train queries, they never entered
the fusion weight or dev, and the epoch count was fixed in advance, so no
selection used them; but the split cannot be published. `train_ce.py` now takes
`--dump-val-qids/--val-qids` so any rerun records its split.

## What we learned (short version)

1. **The official BM25 baseline is not Lucene BM25.** The starter kit applies
   `gensim`'s `LuceneBM25Model` to the query as well, so the dot product carries
   idf twice (plus a query-length normaliser). This explains the published
   numbers exactly (Bitcoin dev 0.0370, IOTA dev 0.3309, macro 0.1147), while a
   real Anserini index on the same corpus gives 0.0300 for Bitcoin. See
   `results/BASELINE_FORENSICS.md`.
2. **Dense is worth ~2.5–3.3× the published BM25 macro** (bge 0.2486, mxbai
   0.2942, two-view 0.3176 over 13 domains, vs ≈0.0967 for the official BM25
   macro on the same 12 domains it covers).
3. **CE-dominant fusion is a consistent but small win** (+0.031 query-level on
   13 domains with mxbai, positive in 9/11 shared domains) but the domain-clustered
   CI crosses zero, so it is not established beyond dev. It fails on `law` and
   `workplace` (CE alone is worse than dense there), so a domain gate is needed
   before using it blindly.
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
results/*.md                  per-phase reports (14 documents)
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
    --out runs/twoview_<domain>_dev.trec --wa 1 --wb 4 --c 60     --json results/fuse_twoview_<domain>_dev.json
PYTHONUTF8=1 python scripts/score_run.py --run runs/twoview_<domain>_dev.trec \
    --data data/track1_tempo/<domain> --split dev --json results/score_twoview_<domain>_dev.json

# 5. cross-domain analysis (bootstrap, fusion weights, LOOCV)
PYTHONUTF8=1 python scripts/analyze_phase7.py --fusions "2,1;1,1;1,2;1,3;1,4"
```

# 6. Phase 8: stronger first stage (mxbai) + CE fusion
python scripts/run_dense_generic.py --index-data data/track1_tempo/<domain> --split dev     --model models/mxbai-embed-large-v1 --tag mxbai --pooling mean     --query-prefix "Represent this sentence for searching relevant passages: "     --work emb --out runs/dense_mxbai_<domain>_dev.trec     --meta runs/dense_mxbai_<domain>_dev.meta.json

python scripts/run_chunk_rerank.py --domain <domain> --split dev     --run runs/dense_mxbai_<domain>_dev.trec --k 100 --model models/bge-reranker-v2-m3     --q-mode title --q-cap 256 --win 250 --pool max --batch 32     --out runs/rerank_mxbai_<domain>_dev.trec     --meta results/rerank_mxbai_<domain>_dev.meta.json

# fusion 1:1 + scoring, then macro/bootstrap over 12 domains
PYTHONUTF8=1 python scripts/analyze_mxbai.py

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

1. **Stronger retrievers: partially addressed.** e5-base (0.148 macro) and
   mxbai (0.2995) are now measured on the same dev split, but the models behind
   the TEMPO numbers (DiVeR-4B, ReasonIR-8B, e5-mistral-7B) do not fit in 8 GB,
   and gte-large is incompatible with transformers 5.22 here. Those numbers
   remain unverified on RETECO v1.1.
2. **Temporal reasoning is not demonstrated, and the benchmark turns out to be
   hostile to it.** Hard-negative mining yields ~0.4 pairs per anchored query;
   demoting year-mismatched documents hurts nDCG in most domains because gold
   documents are frequently "time-mismatched" by content-year heuristics. See
   `results/TEMPORAL_CEILING.md`.
3. **History is squashed.** The published repository is a single curated commit;
   it does not preserve development history. All raw artifacts (runs, scores,
   logs) are included so the numbers remain checkable.

## Data & license

Data and baselines are from the RETECO organizers
(`DataScience-UIBK/RETECO-SemEval2027`, `castorini/prebuilt-indexes-reteco`).
`scripts/ir_metrics.py` is a vendored copy of the official starter-kit scorer
(see its header); the rest of the starter kit is **not** vendored here — see
<https://github.com/DataScienceUIBK/RETECO>.
