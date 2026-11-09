# Dense–cross-encoder fusion for RETECO Track 1a, and a negative result on explicit temporal modelling

System description draft for SemEval-2027 Task 1 (RETECO), Track 1a.
All numbers are dev-split results recomputed from committed TREC runs and qrels;
the hidden test set has not been used. Each number carries the file it was read
from in an HTML comment. `[TODO: verify]` marks a claim not confirmed in the
repository.

---

## Abstract

RETECO Track 1a asks for documents that are both relevant to a query and
consistent with the temporal conditions the query implies. Our system is a
two-stage ranker: a zero-shot dense bi-encoder (`mxbai-embed-large-v1`, full
query) returns 100 candidates, a windowed cross-encoder (`bge-reranker-v2-m3`,
title-form query, 250-token windows with stride 192, max-pooled) re-scores them,
and the two score lists are combined after per-query z-normalisation at a 1:1
weight fixed on the training split. On the 13-domain dev set (519 queries) the
frozen system scores 0.3581 domain-macro nDCG@10, against 0.2943 for the dense
first stage and 0.1147 for the organisers' BM25 baseline
<!-- results/aggregate_sub_sft_e2bc_dev.json; results/phase8_mxbai_summary.json; github_repo/README.md -->.
Swapping the zero-shot cross-encoder for one fine-tuned on the 1,211 train
queries adds +0.0332 (query bootstrap CI [+0.0161,+0.0523]; domain-clustered
[+0.0192,+0.0472]) — the largest single-factor effect in this paper, and one
whose domain-clustered interval excludes zero
<!-- results/boot_dom_e2_vs_zs.json -->. A shorter 1-epoch fine-tuning schedule
adds only +0.0087 under the same fusion, so we report both arms and use the
2-epoch one as the frozen system. A second training seed reproduces the arm at
0.3527 (+0.0278 over the same baseline, domain CI [+0.0159,+0.0404])
<!-- results/REPLICATION.md -->.
We additionally report ten probes of explicit temporal modelling. Exactly one
followed a frozen, pre-registered protocol; it gained +0.0039 (pooled over its
152 anchored queries in 7 domains) against a +0.02 bar
<!-- results/TEMPORAL_GATE.md -->. An annotation-free ceiling analysis explains
the family of negative results: ordering the frozen candidates by year
consistency, with an oracle that knows the query interval exactly, *lowers*
nDCG@10 in 7 of 8 anchored domains <!-- results/TEMPORAL_CEILING.md -->.

## 1 Introduction

Two questions motivate this entry.

**Q1 — how far does a zero-shot pipeline go on this benchmark?** RETECO Track 1a
supplies 13 domain corpora, long natural-language queries and binary qrels, but
no temporal annotations and no document timestamps. A zero-shot dense retriever
plus a zero-shot cross-encoder is therefore the honest starting point, and its
dev score bounds what "no temporal modelling" buys: 0.2943 for dense and 0.3249
fused <!-- results/phase8_mxbai_summary.json; results/aggregate_zs_z11_dev.json -->.

**Q2 — does explicit temporal modelling add anything?** The track's framing
suggests it should. We tested ten ways of injecting time: year and anchor
features, a semantic time-consistency score, oracle and LLM query decomposition,
HyDE, temporal hard-negative mining, a time-aware constraint rerank, a
train-fitted gate, an annotation-free ceiling analysis, and a coverage
diagnostic. One was pre-registered in the strict sense; none produced a gain we
would deploy. Section 5.4 gives the protocol and results, and Section 5.3
explains why they fail: in this benchmark a document that mentions another period
is very often still the right answer.

**Contributions.** (i) A reproducible baseline ladder over all 13 domains under
the task's official aggregation, with paired bootstraps over queries and over
domains. (ii) A supervised cross-encoder that improves the strongest zero-shot
baseline by +0.0332 (and by +0.0087 with a shorter schedule), shown not to be an
artefact of the fusion weight (Section 5.2). (iii) Negative evidence on explicit temporal modelling, with one
pre-registered test and an annotation-free ceiling, plus a correction of two
earlier internal overclaims (the "ten pre-registered negatives" framing and the
reporting statistic; Sections 5.4 and 2.1).

We do not claim that temporal signals are useless in retrieval. Temporal
rerankers with passage timestamps operate on information this track does not
provide; what we show is that *content-time consistency, as extractable from
document text*, did not help under our protocol.

## 2 Task and data

### 2.1 Metric and the missing-query convention

The official aggregation is **domain-macro nDCG@10**: nDCG@10 is computed per
query, averaged within each domain, then averaged over the 13 domains, so every
domain counts equally, and queries missing from a submitted run receive zero
<!-- https://datascienceuibk.github.io/RETECO/evaluation.html, quoted in github_repo/README.md:47-51 and docs/RESEARCH_LOG.md §5i -->.
`scripts/aggregate_scores.py` implements this statistic (domain-macro primary,
topic-macro over all 519 queries as a diagnostic) and now asserts exactly 13
domains <!-- scripts/aggregate_scores.py -->. Zero-filling happens at scoring
time: `scripts/score_run.py` inserts every qrels topic missing from a run with
nDCG@10 = 0 and records `missing_counted_as_zero: true`, unless
`--allow-missing` is passed <!-- scripts/score_run.py -->. This matters because
27.7% of dev queries receive a system score of exactly zero (26.6% under the
frozen 2-epoch system) <!-- results/DEV_ERROR_ANALYSIS.md; results/score_sub_sft_e2bc_*_dev.json -->, so a large share of the metric is lost to
questions the system does not answer at all.

Uncertainty uses a paired bootstrap over 10,000 resamples (seed 7), reported
twice — resampling queries, and resampling domains as clusters — because with 13
domains the second interval is the one that reflects between-domain
heterogeneity <!-- results/boot_dom_sft_vs_zs_11.json -->. We report both rather
than the more favourable one.

### 2.2 Data

The de-duplicated v1.1 release has 13 independent corpora; a query searches only
its own domain. Totals are 1,167,159 documents, 1,211 train and 519 dev queries
<!-- results/corpus_manifest.json (totals); github_repo/README.md:20 -->. Qrels
are binary (all relevance = 1), so nDCG@10 reduces to discounted precision at
the gold set.

| domain | docs | train q | dev q |
|---|---:|---:|---:|
| bitcoin | 76,706 <!-- results/corpus_manifest.json --> | 70 | 30 |
| cardano | 47,509 | 36 | 15 |
| economics | 74,692 | 58 | 25 |
| genealogy | 136,101 | 80 | 35 |
| history | 200,252 | 561 | 240 |
| hsm | 169,552 | 105 | 45 |
| iota | 9,680 | 7 | 3 |
| law | 36,110 | 24 | 11 |
| monero | 53,955 | 46 | 19 |
| politics | 135,207 | 105 | 45 |
| quant | 25,040 | 24 | 10 |
| travel | 158,771 | 70 | 30 |
| workplace | 43,584 | 25 | 11 |

*Table 1. Per-domain corpus and query counts, all from
`results/corpus_manifest.json` (row counts of `documents.jsonl`,
`examples_train.jsonl`, `examples_dev.jsonl`); the columns sum to
1,167,159 / 1,211 / 519.*

Domain sizes differ by more than an order of magnitude and query counts by more
than two (history: 240 dev queries; iota: 3). Under domain-macro aggregation a
3-query domain weighs as much as a 240-query one; Section 7 lists this as a
limitation, and Section 4 checks that the main comparison survives dropping the
smallest domains.

## 3 System

### 3.1 Dense first stage

`mixedbread-ai/mxbai-embed-large-v1` (670,328,392 bytes, 24 layers
<!-- results/MODEL_MANIFEST.json -->), mean pooling, cosine similarity, max
length 512, fp16, query instruction
`Represent this sentence for searching relevant passages: `
<!-- github_repo/README.md:33-39 -->. The dense stage uses the **full** query:
shortening it to the title or the guidance summary hurts dense retrieval in 5 of
6 measured domain/split cells (bitcoin dev 0.2152 full vs 0.1543 title vs 0.1230
summary) <!-- results/QUERY_REPRESENTATION.md -->. This is the opposite of what
the cross-encoder wants: RETECO queries are long forum posts (median 221 tokens,
max 1,017) and are out-of-distribution for MS-MARCO-style re-rankers
<!-- results/RERANK_NOTE.md -->.

### 3.2 Windowed cross-encoder

`BAAI/bge-reranker-v2-m3` (2,271,071,852 bytes
<!-- results/MODEL_MANIFEST.json -->) re-scores the dense top-100. Documents are
split into 250-token windows with stride 192, at most 3 windows per document,
and the document score is the maximum over windows. The query is reduced to its
title form — text before the first `<p>` — capped at 256 tokens
<!-- github_repo/README.md:39-46; scripts/reproduce_all.sh -->. Windowing keeps
the document side from being silently truncated: the full pair in a 512-token
encoder cut documents to roughly 250 tokens on average and scored 0.1577 on
bitcoin dev, while the title query with a 250-token window scored 0.2121 against
a dense baseline of 0.2171 <!-- results/RERANK_NOTE.md -->. Title-form queries
lift every re-ranker we tried (bge-base +0.038, MiniLM +0.034, v2-m3 +0.054)
<!-- results/RERANK_NOTE.md -->.

### 3.3 Score-level fusion

Rank fusion discards score magnitudes, which taxes the stronger component (dense
0.2943 vs zero-shot cross-encoder 0.2680). We fuse z-normalised scores instead,
computed within a query over the intersection of the two runs — the frozen
candidate list, so recall is unchanged and no qrels are read:

    s(d) = 1.0 · z(score_dense(d)) + 1.0 · z(score_CE(d))

<!-- scripts/fuse_scores.py (docstring, --norm z) -->. The 1:1 weight was fitted
once on the 1,211 train queries under the official domain-macro statistic (train
0.2612 at 1:1 vs 0.2596 at 1:2) and frozen; it has never been tuned on dev
<!-- docs/RESEARCH_LOG.md §5i; github_repo/README.md:240-243 -->. An earlier
internal release used RRF with c = 60 at 1:1 (our "V1", dev 0.3166
<!-- results/aggregate_sub_1_1_dev.json -->); we keep it as a reference point
because the change of fusion form is one of the two components of the final
improvement (Section 4).

### 3.4 Supervised fine-tuning

The 1,211 train queries yield 1,071 usable candidate groups, 2,876 positives and
18,544 negatives; 70% of negatives are the highest-scoring dense non-gold
candidates from the top-100 <!-- results/ce_ft/TRAINING_CONFIGS.md -->. Training
is listwise softmax over each candidate group, batch = 1 group, bf16 with
gradient checkpointing, AdamW, 8% warm-up, weight decay 0.01, seed 17, **8 of the
24 layers frozen** (202.6M of 567.8M parameters trainable)
<!-- results/ce_ft/TRAINING_CONFIGS.md -->. We report two schedules: one epoch at
learning rate 1e-5, and two epochs at 1.5e-5, where the second epoch's val nDCG
falls slightly (0.3029 → 0.3012) so the epoch-1 checkpoint is kept. The 2-epoch
schedule is the frozen system. Holding out 45 train queries for monitoring
(never for selection) moves validation nDCG@10 from 0.2870 to 0.3008 (1 epoch)
and from 0.2750 to 0.3029 (2 epochs), AUC from 0.7950 to 0.7990 and 0.7843 to
0.7909 respectively <!-- results/ce_ft/TRAINING_CONFIGS.md -->. Training peaks at
5.5 GB on one 8 GB GPU <!-- github_repo/README.md:253-258 -->.

## 4 Results

| system (dev, 519 queries) | domain-macro (official) | topic-macro (diagnostic) |
|---|---:|---:|
| official BM25 (organisers) | 0.1147 <!-- github_repo/README.md:61 --> | — |
| dense mxbai, top-100 | 0.2943 <!-- results/phase8_mxbai_summary.json --> | 0.2891 <!-- docs/RESEARCH_LOG.md §5i --> |
| zero-shot CE alone | 0.2680 <!-- results/aggregate_zs_ce_dev.json --> | 0.2990 <!-- results/aggregate_zs_ce_dev.json --> |
| SFT CE alone (1 epoch) | 0.2742 <!-- results/aggregate_sft_ce_dev.json --> | 0.3096 <!-- results/aggregate_sft_ce_dev.json --> |
| SFT CE alone (2 epochs) | 0.2823 <!-- re-scored runs/rerank_sft_e2bc_*_dev.trec; see results/PAPER_VERIFY_E2.md --> | — |
| dense + zero-shot CE, RRF 1:1 (V1) | 0.3166 <!-- results/aggregate_sub_1_1_dev.json --> | 0.3199 <!-- results/aggregate_sub_1_1_dev.json --> |
| dense + zero-shot CE, z 1:1 | 0.3249 <!-- results/aggregate_zs_z11_dev.json --> | 0.3260 <!-- results/aggregate_zs_z11_dev.json --> |
| dense + SFT CE (1 epoch, lr 1e-5), z 1:1 | 0.3336 <!-- results/aggregate_sub_sft11_dev.json --> | 0.3339 <!-- results/aggregate_sub_sft11_dev.json --> |
| **dense + SFT CE (2 epochs, lr 1.5e-5), z 1:1 (frozen)** | **0.3581** <!-- results/aggregate_sub_sft_e2bc_dev.json --> | 0.3644 <!-- results/aggregate_sub_sft_e2bc_dev.json --> |
| dense + SFT CE (2 epochs), seed 29 (replication) | 0.3527 <!-- results/aggregate_sub_sft_e2s29_dev.json --> | 0.3624 <!-- results/aggregate_sub_sft_e2s29_dev.json --> |

*Table 2. Dev results. The BM25 row is the organisers' baseline measured with
their code on the v1.1 dev split (0.1147; the TEMPO paper reports 0.108 on the
full set, and the earlier 12-domain v1.0 macro is ≈0.0967
<!-- scripts/official/BASELINE_RESULTS.md:3-16 -->), so it is a quoted reference
rather than a system we tuned. The repo prints 0.2942 for the dense stage in
three places; the summary JSON holds 0.2942754, which we round to 0.2943.*

| contrast (paired bootstrap, 10k, seed 7) | Δ domain-macro | query CI | domain CI |
|---|---:|---|---|
| SFT CE vs zero-shot CE, fixed z 1:1 (2-epoch arm, frozen) | +0.0332 <!-- results/boot_dom_e2_vs_zs.json --> | [+0.0161,+0.0523] <!-- results/boot_dom_e2_vs_zs.json --> | [+0.0192,+0.0472] <!-- results/boot_dom_e2_vs_zs.json --> |
| SFT CE vs zero-shot CE, fixed z 1:1 (2-epoch arm, seed 29) | +0.0278 <!-- results/boot_dom_e2s29_vs_zs.json --> | [+0.0118,+0.0455] <!-- results/boot_dom_e2s29_vs_zs.json --> | [+0.0159,+0.0404] <!-- results/boot_dom_e2s29_vs_zs.json --> |
| SFT CE vs zero-shot CE, fixed z 1:1 (1-epoch arm) | +0.0087 <!-- results/boot_dom_sft_vs_zs_11.json --> | [+0.0025,+0.0151] <!-- results/boot_dom_sft_vs_zs_11.json --> | [+0.0037,+0.0142] <!-- results/boot_dom_sft_vs_zs_11.json --> |
| z 1:1 zero-shot vs V1 RRF (fusion form) | +0.0083 <!-- results/boot_dom_z11_vs_v1.json --> | [−0.0010,+0.0179] <!-- results/boot_dom_z11_vs_v1.json --> | [−0.0015,+0.0188] <!-- results/boot_dom_z11_vs_v1.json --> |
| SFT z 1:1 vs V1 RRF (total, 1-epoch arm) | +0.0170 <!-- results/boot_dom_sft_vs_v1.json --> | [+0.0055,+0.0290] <!-- results/boot_dom_sft_vs_v1.json --> | [+0.0052,+0.0287] <!-- results/boot_dom_sft_vs_v1.json --> |

*Table 3. Inferential claims. For the 1-epoch arm the total over V1 is exactly the
sum of its two components (0.0083 + 0.0087 = 0.0170), but only the fine-tuning
component excludes zero at the domain level; we attribute +0.0087 to supervision
and treat the fusion-form change as unestablished. The 2-epoch arm is compared
only against the zero-shot system at the same frozen fusion.*

The frozen 2-epoch system wins 188 dev queries and loses 78 against the zero-shot
fused baseline, with 253 ties <!-- results/boot_dom_e2_vs_zs.json -->, and 11 of
13 domains move up (`hsm` +0.071, `law` +0.076, `workplace` +0.063, `cardano`
+0.055; only `bitcoin` −0.002 and `quant` −0.001 fall)
<!-- results/WEIGHT_SWEEP_E2.md -->. Dropping iota (3 queries) leaves the
comparison at 0.3181 vs 0.2832; restricting to the 8 domains with at least 19
queries gives 0.3267 vs 0.2985 <!-- results/ROBUSTNESS.md -->. The gain is not
carried by the smallest domains.

A second training seed (29, same recipe and checkpoint-selection rule) gives
0.3527 domain-macro, i.e. +0.0278 over the same zero-shot baseline with a
domain-clustered interval of [+0.0159,+0.0404], 181 wins / 82 losses / 256 ties,
and 0.3123 vs 0.2832 on the 12-domain view
<!-- results/REPLICATION.md; results/boot_dom_e2s29_vs_zs.json -->. Seed-to-seed
spread of the arm (0.0054) is an order of magnitude below the delta over the
baseline, so we report the effect as **+0.031 ± 0.003 over the strongest
zero-shot baseline** and keep 0.3581 as the frozen submission number.

The pre-registered adoption bar — dev topic-macro at least V1 + 0.015 — was
**not** met by the 1-epoch arm under the statistic registered at the time:
SFT − V1 = 0.33393 − 0.31987 = +0.0141 <!-- github_repo/README.md:275-279 -->. We
report this rather than restating the bar under the corrected metric.

## 5 Analysis

### 5.1 Why rerank at all, if the cross-encoder alone is weaker?

The cross-encoder alone scores 0.2742 (1-epoch) or 0.2823 (2-epoch), below the
dense retriever's 0.2943, yet fusion reaches 0.3581. Three observations make this
coherent. First, re-ranking
is a *replacement* operator, so its damage is concentrated where nDCG is
computed: with a poor re-ranker, replacing the top-5 costs −0.003 but replacing
the top-100 costs −0.098 <!-- results/RERANK_NOTE.md -->. A model slightly worse
on the full list can be much worse at the head, and vice versa. Second, the
errors are largely complementary: the mean within-query Spearman correlation
between dense and SFT cross-encoder scores over the top-100 is 0.328 (0.279 for
the 2-epoch arm)<!-- results/weight_sweep_dev.json; results/weight_sweep_e2_dev.json -->, so a linear combination can order
candidates better than either component. Third, the two stages discriminate
about equally well within a query (AUC 0.806 dense vs 0.801 zero-shot CE)
<!-- results/PHASE9_PLAN.md -->; what differs is *which* candidates they rank
high. Fusion is therefore a variance-reduction device, bounded by the headroom
above the retriever: the oracle over the dense top-100 is 0.7446
<!-- github_repo/README.md:219-227 -->, of which the frozen system captures 14.2%
<!-- results/CAPTURE_CURVE_E2.md; results/capture_curve_e2.json -->. Much of the remaining loss is not ordering
at all: 26.6% of dev queries score zero, and 37.7% of those have no gold document
in the top-100 <!-- results/score_sub_sft_e2bc_*_dev.json -->.

### 5.2 How much does the fusion weight matter?

| w (CE) | zero-shot | SFT | Δ | domain CI | domains up |
|---:|---:|---:|---:|---|---:|
| 0.5 | 0.3208 | 0.3211 | +0.0004 | [−0.0031,+0.0042] | 5/13 |
| 0.7 | 0.3294 | 0.3287 | −0.0007 | [−0.0081,+0.0053] | 7/13 |
| 0.85 | 0.3261 | 0.3308 | +0.0047 | [+0.0001,+0.0098] | 8/13 |
| **1.0 (frozen weight)** | 0.3249 | **0.3336** | **+0.0087** | [+0.0038,+0.0142] | 10/13 |
| 1.15 | 0.3271 | 0.3319 | +0.0048 | [+0.0008,+0.0089] | 9/13 |
| 1.3 | 0.3278 | 0.3334 | +0.0056 | [+0.0004,+0.0114] | 10/13 |
| 1.5 | 0.3253 | 0.3298 | +0.0045 | [−0.0018,+0.0108] | 8/13 |
| 1.8 | 0.3192 | 0.3341 | +0.0149 | [+0.0055,+0.0293] | 12/13 |
| 2.0 | 0.3193 | 0.3274 | +0.0080 | [+0.0041,+0.0122] | 11/13 |

*Table 4. Diagnostic sweep of the fusion weight on dev for the 1-epoch arm; nothing is selected
here. All cells from `results/WEIGHT_SWEEP.md` / `results/weight_sweep_dev.json`
(domain-clustered bootstrap CIs). The sweep uses 2,000 resamples, so the w = 1
interval's last digit differs from the 10,000-resample interval in Table 3.*

Two things are true at once, and we state both. The sweep shows the effect does
not hinge on a lucky weight: the SFT arm is ahead at every w ≥ 0.85, and only
w = 0.7 is negative (−0.0007) <!-- results/WEIGHT_SWEEP.md -->. But the curve is
also *flat* to within about ±0.015 with no monotone trend, and the zero-shot
arm's own movement across the sweep (0.3192–0.3294) is the same size as the
effect being measured <!-- results/WEIGHT_SWEEP.md -->. These differences are
mostly reranking noise, not weight sensitivity. We therefore claim the
frozen-weight number only, and we note the direction of the bias: the weight was
fitted with the *zero-shot* arm, so the supervised arm is the under-tuned one and
+0.0087 is, if anything, conservative
<!-- github_repo/README.md:283-289 -->.

The same sweep for the 2-epoch arm is always positive (and wider than the
1-epoch one, so it is not evidence of a flat surface): the domain
delta is +0.0089 at w = 0.5, +0.0332 at the frozen w = 1, +0.0339 at 1.5 and
+0.0346 at 1.8, and the SFT arm is ahead at all nine weights
(`results/WEIGHT_SWEEP_E2.md`). At w = 0.7, 1.5 and 1.8 all 13 domains improve.
This removes the "lucky weight" concern for the headline system: even a weight
that grossly over-weights the cross-encoder keeps the gain.

### 5.3 The temporal ceiling: year consistency is not a relevance signal

Before blaming any particular temporal module, we asked what the best possible
year-consistency reranker could add. For every dev query with a parseable year
interval we reordered the frozen dense top-100 using only the query interval and
the document's extracted year — no qrels — with two oracles: `perfect`
(consistent documents first) and `demote` (push inconsistent documents to the
end, the realistic variant since most documents have no extractable year)
<!-- results/TEMPORAL_CEILING.md -->.

On the 169 anchored queries of 8 domains the dense macro is 0.2980 (over those 8
domains, not the full 13); `perfect` gives 0.2498 and `demote` 0.2834
<!-- results/TEMPORAL_CEILING.md -->. The oracle ordering lowers nDCG@10 in 7 of
8 domains and the realistic variant in 6 of 8; economics is the only clear winner
(0.1890 → 0.1982) and workplace is flat (+0.0009)
<!-- results/TEMPORAL_CEILING.md -->. The mechanism is visible in the mismatch
rate: among candidates with an extractable year, 57–100% lie outside the query
interval, and many of them are gold <!-- results/TEMPORAL_CEILING.md -->. A
document that discusses another period — as background, comparison or
retrospective — is frequently the right answer to a question about a period.
"Time mismatch" is thus not a usable negative signal here, and an upper bound
that assumes it is cannot be beaten by a better extractor of the same signal.

Two caveats keep this honest: the ceiling is computed over *reorderings of the
frozen candidate set*, so it says nothing about recall, and it covers 169 of 519
dev queries in 8 domains <!-- results/TEMPORAL_CEILING.md -->.

### 5.4 Ten temporal probes: protocol and results

**Protocol.** No test-set access. All probes run on dev only, after a design
fixed in advance; where a parameter or threshold was fitted, it was fitted on
train — except the dev-fitted oracle in row 3, labelled as such, and row 2's
cross-validation, a feasibility check with no deployed model. Every probe is
reported whatever its outcome. In the strict sense used in our notes — frozen
protocol, pre-registered adoption bar, dev read once — **exactly one probe was
pre-registered** (row 9). Rows 1–2 fixed their designs and thresholds before
running but were not registered <!-- results/TIME_ABLATION.md;
results/TEMPORAL_SEMANTIC_PROBE.md -->; rows 3–8 and 10 are exploratory, and
only row 9 has a frozen protocol <!-- results/NEXT_STEPS.md:84-90 -->. Earlier
internal drafts described all ten as "pre-registered negatives"; that framing
was wrong and is corrected here.

| # | probe | what was tested | protocol | dev result |
|---:|---|---|---|---|
| 1 | year/anchor feature ablation | LR over year, temporal anchors and guidance flags on a bge-base dense top-100, vs length/lexical controls | fit on train, dev once | bitcoin −0.0230 [−0.0651,+0.0078]; economics +0.0232 [−0.0254,+0.0773]; controls hurt both <!-- results/TIME_ABLATION.md --> |
| 2 | semantic time-consistency score | cosine between a time-statement built from guidance and each candidate | 5-fold query-grouped CV; placebo control; pre-set bar ΔAUC ≥ 0.05–0.08 | ΔAUC +0.00005 / +0.0065; pooled ΔAUC −0.0062 / +0.0008 <!-- results/TEMPORAL_SEMANTIC_PROBE.md --> |
| 3 | oracle step rerank | guidance-derived temporal steps fused with dense | per-domain weight chosen on dev (oracle) | bitcoin 0.2393 at w0=4 (dense 0.2171); economics 0.1652 at w0=8 (dense 0.1891) <!-- results/TEMPORAL_DECOMPOSITION.md --> |
| 4 | LLM step decomposition | Qwen2.5-1.5B rewrites the query into temporal steps | 35 calls/query; w0 = 4 reported | 0.2012 / 0.1613, both below dense <!-- results/TEMPORAL_DECOMPOSITION.md --> |
| 5 | HyDE | 5 hypothetical documents per query, α = 1.0 | fixed α | 0.1819 / 0.1529, both below dense <!-- docs/RESEARCH_LOG.md §5f --> |
| 6 | temporal hard-negative mining | year-inconsistent candidates as training negatives | mined candidates | 37–43 across five domains (≈0.4 per anchored query): too few to train <!-- results/TEMPORAL_CEILING.md --> |
| 7 | naive TACR rerank | time-aware constraint reordering | no frozen protocol | ranking collapse; no metric retained <!-- results/TEMPORAL_CEILING.md --> |
| 8 | year-consistency ceiling | perfect / demote reordering of the frozen top-100 | annotation-free; 169 anchored queries, 8 domains | macro 0.2498 / 0.2834 vs 0.2980; worse in 7/8 and 6/8 <!-- results/TEMPORAL_CEILING.md --> |
| 9 | temporal gate | train-fitted gate over temporal features on the dense top-100 (CE scores as features) | **pre-registered**: fit on train, λ inside train, dev once, bar +0.02 | pooled over 152 anchored queries in 7 domains: 0.3862 → 0.3901, Δ+0.0039, query CI [−0.0023,+0.0103], domain CI [−0.0006,+0.0142], P(Δ>0)=0.893 <!-- results/TEMPORAL_GATE.md --> |
| 10 | temporal observability | share of documents/queries with an extractable year | diagnostic, no ranking | docs 20.4–50.6%; queries 6.7–57.8% <!-- results/TEMPORAL_OBSERVABILITY.md --> |

*Table 5. Ten probes. Rows 3–5 are oracle or exploratory probes, not deployed
systems; row 7 left no metric behind and is reported as a qualitative failure.*

What this licenses: on RETECO Track 1a, with no timestamps and only text-derived
time, none of these probes produced a gain we would submit, and the one tested
under a frozen protocol fell below its own bar by a factor of five. What it does
not license: the claim that temporal modelling is useless. It also does not show
that our features are the best possible content-time representation — only that
several independent families (lexical, semantic, generative, constraint-based)
failed together, and that the ceiling analysis predicts they would.

## 6 Efficiency and reproducibility

**Runtime.** Cross-encoding the dense top-100 on the full dev set (519 queries)
took 3,086 s ≈ 51 min of GPU time for 240,554 document windows, about 5.95 s per
query, peaking at 1,308 MB
<!-- results/rerank_mxbai_*_dev.meta.json (13 files, `seconds` summed) -->. The
end-to-end script, including dense retrieval, fusion, de-tying, scoring and
aggregation, is documented at ≈70 min for all 13 domains and ≈2 min for one
domain on a single RTX 4060 <!-- scripts/reproduce_all.sh:1-11 -->. Document
embeddings are cached per (model tag, corpus size) and reused, so the 70-minute
figure assumes warm caches; a cold run must encode 1,167,159 documents, which at
the measured ≈46 documents/s in travel, the most expensive domain (158,771
documents), is a multi-hour job
<!-- scripts/run_dense_generic.py:95-106; results/PHASE8_STRONGER_BASELINES.md:146-148 -->.

**One-command reproduction.**

```bash
bash scripts/reproduce_all.sh                                # 13 dev domains, zero-shot CE
CKPT=results/ce_ft/<run>/best bash scripts/reproduce_all.sh  # supervised CE
DOMAINS=iota bash scripts/reproduce_all.sh                   # one domain, ~2 min
```

The script runs dense top-100 → cross-encoder (`--q-mode title --q-cap 256
--win 250 --stride 192 --pool max`) → z-score fusion 1:1 → de-tied validation with
the official checker → scoring → aggregation <!-- scripts/reproduce_all.sh -->.

**Hashes and released material.**

| artifact | identifier |
|---|---|
| `mxbai-embed-large-v1` weights | sha256 `36bfa45da00eb762ef9feebe4cff315ec779efadd08da11846bef5ba5b59b8f8` <!-- results/MODEL_MANIFEST.json --> |
| `bge-reranker-v2-m3` weights | sha256 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286` <!-- results/MODEL_MANIFEST.json --> |
| corpus manifest | 13 domains, 65 files, per-file sha256 and row counts (e.g. bitcoin `documents.jsonl` `8e04403571083f1fbb92a27ca42641cff5d343782b9c1b40be86e92591952795`) <!-- results/corpus_manifest.json --> |
| submitted runs | 39 TREC files hashed in `github_repo/RUN_MANIFEST.sha256` <!-- github_repo/RUN_MANIFEST.sha256 --> |
| environment | Python 3.12.7, torch 2.7.0+cu126, transformers 5.22.2 <!-- results/rerank_bge_bitcoin_dev_k100.meta.json -->, one 8 GB GPU |

Released: all code, the 13 fused dev TREC runs, per-query score JSONs, aggregate
JSONs, the corpus and model manifests, and the experiment log. Not released: the
corpora and prebuilt indices (licensing and size), the model weights, and the
2.2 GB fine-tuned checkpoints
<!-- results/ce_ft/TRAINING_CONFIGS.md; github_repo/README.md -->.

## 7 Limitations and ethics

**Two training seeds, small effect.** The frozen arm was replicated with a
second seed (29): 0.3527 vs 0.3581 domain-macro, delta over the zero-shot
baseline +0.0278 (domain CI [+0.0159,+0.0404]) vs +0.0332
<!-- results/REPLICATION.md -->. The bootstrap intervals resample queries and
domains, not seeds, so with two runs we can bound seed variance only roughly;
with 13 domains a single domain can still swing the macro. We therefore describe
the gain as ~+0.03 and dev-specific, not as a stable property of the method.

**An unrecoverable validation split.** Training held out 45 train queries (seed
17, 12% per domain capped at 4) for monitoring and selected nothing on them.
Replaying that split from the current code does not reproduce the logged
per-domain counts for any of the 13 domains across seeds 0–60, so the exact 45
queries cannot be published. They are train queries; they never entered the
fusion weight or dev, and the epoch count was fixed in advance
<!-- github_repo/README.md:308-315 -->. `train_ce.py` now takes
`--dump-val-qids/--val-qids` so future reruns record their split.

**Dev only.** The hidden test set is untouched. Nothing here is a leaderboard
claim, and the system was frozen before any test evaluation.

**Missing queries count as zero.** This is the official convention and it
penalises incomplete submissions; our frozen runs cover all 519 dev queries and
13/13 files pass the official checker <!-- github_repo/README.md:244-252 -->, so
the rule does not flatter us — but 26.6% of queries still score zero on merit
<!-- results/DEV_ERROR_ANALYSIS.md -->.

**Weight and metric.** The fusion weight was fitted on train with the zero-shot
arm under the corrected domain-macro statistic; an earlier internal release
selected it on dev under topic-macro, which we report as a correction rather than
hide <!-- docs/RESEARCH_LOG.md §5i -->. The pre-registered topic-macro bar was
not met (+0.0141 against +0.015); the inferential claims rest on the corrected
official statistic, a change of estimand after the fact, stated as such.

**Ceiling analysis.** Annotation-free but extractor-dependent: mismatch rates
condition on documents where a year could be parsed, coverage is partial
(20.4–50.6% of documents), and the analysis covers 169 anchored queries in 8
domains <!-- results/TEMPORAL_CEILING.md; results/TEMPORAL_OBSERVABILITY.md -->.
It bounds reorderings of the frozen candidates, not retrieval.

**Heterogeneity.** Domain-macro aggregation gives iota's 3 dev queries the same
weight as history's 240; iota's system score is 0.8383 in isolation
<!-- results/ROBUSTNESS.md -->. Per-domain deltas range from +0.0760 (law) to
−0.0023 (bitcoin) <!-- results/WEIGHT_SWEEP_E2.md -->, so the aggregate hides real
disagreement. The 12-domain and n ≥ 19 checks in Section 4 are the mitigation we
can offer, not a solution.

**Ethics.** The corpora are public forum and wiki-like text released by the task
organisers; we collected no new human annotations, ran no user study, and
processed no private data. Retrieval systems of this kind can surface outdated or
offensive material and can amplify whatever the source community wrote; because
no temporal module is deployed, the frozen system makes no temporal guarantee at
inference and should not be used as if it did. We did not run a bias or toxicity
audit of the ranked outputs, and we do not claim one. Compute was one consumer
GPU; the energy cost was not metered.
