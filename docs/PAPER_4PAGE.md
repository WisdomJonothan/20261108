# Multi-evidence re-ranking for RETECO Track 1a (4-page system description draft)

> Status: draft, numbers frozen as of 2026-11-12 02:30 (loss arms S5.5, step
> arms S5.6, query-representation and recall-backfill arms S5.7). No pending
> cells.
> Companion documents: `docs/PAPER_DRAFT.md` (long version, 520 lines),
> `docs/RESEARCH_PLAN_NEXT.md` (ladder + protocol), `results/PHASE11_VALIDATION.md`.

## Abstract

RETECO Track 1a asks for documents that answer a *complete* question about a
historical period. Queries are long, their temporal and exclusion conditions sit
in the body rather than the title, and the judgements mark **several** documents
per question (3.44 on average) over **2.34 annotated steps**.
We describe a two-stage ranker — a zero-shot dense retriever followed by a
windowed cross-encoder fine-tuned on the 1,071 training queries that carry at
least one positive among their dense top-100 candidates — that scores
0.3581 domain-macro nDCG@10 on the 13-domain dev set, against 0.2943 for the
dense first stage and 0.1147 for the organisers' BM25 baseline. The supervised
reranker contributes +0.0332 (domain-clustered 95% CI [+0.0192, +0.0472]) at a
fusion form fixed before fine-tuning, and reproduces at +0.0326 / +0.0279 under
two further seeds. An analysis of the frozen system shows the remaining headroom
is only partly recall: at the 100 candidates the reranker sees, 80.7% of annotated
steps have at least one relevant document (96.3% within the dense top-1000), yet
the fused top ten covers only 59.5% of the annotated steps and the cross-encoder's
own top ten 51.4%, and only 38.3% of questions place two relevant documents there. Explicit temporal
modelling did not close this gap in ten probes, including one strictly
pre-registered test (+0.0039 against a +0.02 bar) and a fixed year-consistency
reordering that *lowers* nDCG@10 in 7 of 8 domains.
Supervising the cross-encoder with the organisers' step annotations -- so that
every step must have a positive ranked high -- costs 0.014 domain-macro on dev
and *lowers* step coverage, 59.5% to 55.6%/56.0% in the fused list and 51.4% to
48.1%/45.8% for the cross-encoder alone. Two further corrections, a
constraint-faithful 64-word full-question query and the restoration of the 140
training queries whose dense top-100 contains no positive, each cost 0.0138 on
the same split - but only 0.0026 against a same-code retrain of the unchanged
objective, which itself costs 0.0111, so on dev they are indistinguishable from
it. Every intervention in the six-step ladder is neutral or negative on dev,
and the frozen system is kept.

## 1 Introduction

Track 1a gives 13 independent corpora, long queries, binary qrels and no
timestamps. Our entry has two parts: a strong, reproducible zero-shot-to-supervised
pipeline, and a diagnosis of why the obvious temporal fixes do not work.

Contributions: (i) a baseline ladder over all 13 domains under the official
domain-macro statistic, with paired bootstraps over queries and over domains;
(ii) a supervised cross-encoder that adds +0.0332 without tuning the fusion
weight on dev; (iii) a multi-evidence analysis showing that the system's failure
mode is "one easy document" inside a candidate pool that is itself incomplete;
(iv) negative evidence on explicit temporal modelling: one pre-registered test
and a fixed year-consistency rule that fails.

*Relation to prior work.* The reranker is the standard retrieve-then-rerank
cross-encoder (Nogueira and Cho, 2019), initialised from BGE M3 (Chen et al.,
2024). Our multi-evidence objective is a sibling of listwise and LambdaRank
training (Burges, 2010; Wang et al., 2018), with one difference: instead of
ordering one list, it asks that every official step of a question own a document
in the top ten. Temporal retrieval is the benchmark's own subject (Abdallah et
al., 2026); our contribution there is negative evidence about what does not
transfer from the two-domain pilot to the full suite.

## 2 Task, metric, data

**Metric.** Official aggregation is domain-macro nDCG@10: per-query nDCG@10,
averaged within domain, then over the 13 domains, so a 3-query domain weighs as
much as a 240-query one, and queries missing from a run score zero. dev = 519
queries. 26.6% of dev questions receive exactly zero from the frozen system, and
37.7% of those have no relevant document in the top 100 at all. Those zeros
dominate the statistic: had the 138 zero questions merely reached half their
domain's average, domain-macro would be 0.4283 instead of 0.3581 (full average:
0.4985, `results/ZERO_HEADROOM.json`).

**Data.** 1,167,159 documents, 1,211 train queries, binary qrels (so nDCG@10 is
discounted precision at 10). Per-domain counts in `results/corpus_manifest.json`.
Queries carry a title and a body separated by `<p>`; the body holds the temporal
range, exclusions and comparisons (Section 5.3).

## 3 System

**Dense first stage.** `mxbai-embed-large-v1`, full query with the model's
retrieval instruction, mean pooling, cosine similarity, fp16, `top-100`
candidates per query. Dev: 0.2943 (0.2956 with CLS pooling: a wash).

**Windowed cross-encoder.** `bge-reranker-v2-m3` (568M). Documents are split into
250-token windows with stride 192, each window is scored against the query, and
the document score is the maximum over its windows, i.e. a document is as
relevant as its best passage; query capped at 256 tokens. Pooling was fixed a
priori and is not ablated (Section 7). Zero-shot at fusion 1:1: 0.3249.

**Fusion.** Per-query z-normalisation of the dense and cross-encoder lists,
combined at 1:1. The weight was fixed on the training split before any dev
inspection; Section 5.1 shows the choice is not what produces the gain.

**Supervised fine-tuning.** Listwise softmax over a 20-document group sampled
from each training query's 100 candidates - 21,420 query-document pairs in
1,071 groups, the log-sum-exp running over the group's positives - 8 of 24
layers frozen, lr 1.5e-5, seed 17, 2 epochs; epoch 1 was kept, by a 0.3029-against-
0.3012 internal-validation margin. Checkpoint selection used the cross-encoder's
own ordering on 45 held-out training queries, not the fused system; that run
predates domain-macro logging and selected on the plain mean, and on every later
arm the two criteria pick the same epoch. Frozen dev: **0.3581**.

**Multi-evidence (step) supervision.** Track 1a gives one relevance judgement per
query, but the organisers' qrels also carry *step* annotations: a question that
needs several pieces of evidence is decomposed into steps, and each step lists
the documents that satisfy it. We use them only in training; at test time the
ranker still sees the full question and one document. With `P_j` the step-`j`
positives inside a query's candidate list, the ordinary listwise term
`-log sum_{d in P} p(d)` is dominated by the single easiest positive, so we add

```
L_step = -(1/m) * sum_j log sum_{d in P_j} p(d),    L = L_list + lambda * L_step
```

which forces *every* step to have a positive ranked high instead of letting one
easy document carry the query. Steps whose positives are absent from the
candidate list are skipped (a step is covered for 91.2% of step instances under
our candidate construction); `m` counts the covered steps. `lambda` is not tuned
on dev: two values (0.3, 1.0) are trained on the same groups, and both raise the
train-internal split above the baseline (0.2882 and 0.2913 against 0.2865), yet
both are rejected on dev (S5.6).

Mechanistically it targets the failure mode of S5.2: steps are retrievable
(80.7% at the 100 candidates the reranker sees, 96.3% at depth 1000) far more
often than they reach the top ten: 59.5% of annotated steps reach it in the fused
list and 51.4% for the cross-encoder alone, with every annotated step that has a
positive in the denominator.

## 4 Results

| system | dev domain-macro nDCG@10 | Δ vs zero-shot z1:1 | domain-clustered 95% CI |
|---|---:|---:|---|
| official BM25 | 0.1147 | — | — |
| dense (mxbai, top-100) | 0.2943 | — | — |
| dense + zero-shot CE, z 1:1 | 0.3249 | — | — |
| + SFT CE, 1 epoch | 0.3336 | +0.0087 | [+0.0037, +0.0142] |
| **+ SFT CE, 2 epochs (frozen)** | **0.3581** | **+0.0332** | **[+0.0192, +0.0472]** |
| same recipe, seed 23 | 0.3575 | +0.0326 | [+0.0184, +0.0470] |
| same recipe, seed 29 (two independent reruns) | 0.3528 / 0.3527 | +0.0279 | [+0.0160, +0.0405] |
| fresh checkpoint, RRF 1:4 (OOF-selected form) | 0.3187 | +0.0205 | [+0.0010, +0.0404]* |
| fresh checkpoint, RRF 1:1 (V1 form) | 0.3431 | +0.0265 | [+0.0052, +0.0473]* |

\* the two RRF rows are compared against zero-shot CE at the *same* form (0.2981 and
0.3166), because an RRF list is not on the z-fusion scale; the 0.3249 reference
does not apply to them. See Section 5.1.

Robustness: the gain holds when the three smallest domains are dropped
(0.3267 vs 0.2985 on n≥19 domains) and in 11 of 13 domains (the two exceptions, bitcoin and quant, are −0.0023 and
−0.0009, i.e. flat). Seed spread over
three seeds is 0.0029, an order of magnitude below the effect; a fourth run of
the same recipe through the current code path scores 0.3470, so the recipe's
retraining band is 0.347–0.358 and the frozen artifact sits at its top (S5.5,
S7).

## 5 Analysis

### 5.1 Is the gain an artefact of the fusion weight?
No. A sweep over nine weights (0.5…2, CE weight) is positive everywhere (+0.0089…+0.0346;
the w=1 row of that sweep reproduces the headline within bootstrap noise),
and refitting the weight out-of-fold on 45 held-out training queries selects a
dense-heavy RRF 1:4 that is *worse* than z 1:1, so the gain is not bought by
tuning on dev.

### 5.2 The bottleneck is multi-evidence coverage over an incomplete pool
We annotate-free measured, per system, how much of the official step structure
reaches the top ten (`scripts/dev_multievidence.py`, `results/DEV_MULTIEVIDENCE.md`):

| system | nDCG@10 | relevant docs in top-10 | questions with ≥2 | step coverage of top-10 | steps retrievable @1000 |
|---|---:|---:|---:|---:|---:|
| dense | 0.2943 | 1.03 | 31.8% | 51.3% | 96.3% |
| + zero-shot CE | 0.3249 | 1.11 | 35.6% | 54.9% | 96.3% |
| + SFT CE (frozen) | 0.3581 | 1.21 | 38.3% | 59.5% | 96.3% |

Supervision improves every column but leaves most of the structure unused. At
depth 1000, seven of ten steps are retrievable yet four of ten never reach the
top ten; at the depth the reranker actually sees - the rerank files keep 100
candidates per query - only 80.7% of annotated steps are reachable and 88.3% of
questions keep at least one positive, so a fifth of the step structure is lost
before the cross-encoder runs. Truncating the fused list from 1000 to 100 rows
leaves its top ten unchanged (59.45%, `results/MULTIEVIDENCE_sft_e2bc_sub100.json`),
so depth moves reachability only; the 51.4% above is the cross-encoder's own
ordering over those 100 candidates, a different list from the fused one. That limit is visible in the zeros: 52 of the 138
questions the frozen system scores at zero (38%) have no relevant document in
the top 100 at all (`results/DEV_ERROR_ANALYSIS.md`), and no reranker over those
candidates can lift them. The retrieval floor proper is small: in 10 of the 519
dev questions (1.9%) no relevant document appears anywhere in the dense
top-1000. Coverage is
not a cosmetic statistic: on the frozen dev run, per-question coverage and nDCG@10
are strongly associated (Spearman 0.69), and questions whose steps are all covered
average 0.595 nDCG@10 against 0.187 otherwise (228 vs 256 questions,
`results/STEP_COVERAGE_CORR.json`). The association is partly mechanical -- finding
a step's documents is how a question earns credit -- so we use it only as a
mechanism check, not as evidence of gain; arms are still judged on dev nDCG.
This is the motivation for multi-evidence (step-aware) ranking supervision;
Section 5.2b shows the present objective does not close it.

### 5.2b Where the gain sits: not on the multi-step questions
Partitioning dev by the number of official evidence steps a question needs
(`results/STEP_SPLIT.md`), the frozen system's gain over zero-shot CE is largest
on single-step questions and weakest on two-step ones:

| partition | queries | zero-shot CE | frozen SFT | delta | 95% CI (domain-clustered) |
|---|---:|---:|---:|---:|---|
| 1 step | 75 | 0.3185 | 0.3896 | +0.0711 | [+0.0244, +0.1056] |
| 2 steps | 151 | 0.3857 | 0.4058 | +0.0200 | [-0.0033, +0.0463] |
| 3+ steps | 258 | 0.3196 | 0.3511 | +0.0315 | [+0.0084, +0.0571] |
| none | 35 | 0.1579 | 0.1705 | +0.0126 | [+0.0030, +0.0524] |

The present objective therefore buys a better *single-document* decision almost
everywhere, but it does not specifically teach evidence assembly: on the 151
two-step questions the gain is within noise. That is exactly the gap `L_step`
targets, and it fixes the bar the step arms have to clear -- a gain that moves the
two-step partition, not only the overall mean.

### 5.3 Query truncation drops the conditions
In 1,730 train+dev queries, 44.8% (46.4% of dev: 241 of 519) have at
least one condition (year 14.3%, temporal relation 19.0%, exclusion 16.3%,
comparison 14.0%) that appears **only in the body**, i.e. it never reaches the
cross-encoder, which is trained on the title alone; the dense stage reads the
full question.

That truncation is not, by itself, what the ranker is missing. Splitting dev by
this flag (`results/CONSTRAINT_SUBSET.md`), the frozen system scores **0.3656 on
the 241 condition-lost queries vs 0.3518 on the rest**, and its gain over
zero-shot CE is about twice as large there (+0.0481 vs +0.0246) although the dense stage reads the full question and only the reranker does
not. A same-budget body-visible
composer therefore has to *earn* the claim: we pre-register that its gain must
concentrate on the lost subset and not come from the kept one. Two 64-word
composers were built - a cue-sentence selector that keeps the title plus only
those body sentences carrying a condition, and the plain full question (title
plus the first 64 body words). On the 45-query internal split no representation
wins outside noise: title 0.2827, full 0.2869, cue-sentence selector 0.2744,
fixed body 0.2703, frequency summary 0.2614 (`results/QVARIANT_WINNER.json`); the
~+-0.03 resolution of that split (S7) makes dev the decisive test. The selector
is the more faithful of the two and it was the worst of the five, so dev carries
the *generous* arm - the whole body is there and no selection step can hide a
mistake. Trained end to end on that 64-word full question, it scores **0.3443**
on dev against 0.3581 for the frozen system, whose reranker reads titles only
(Delta -0.0138,
query-clustered 95% CI [-0.0272, -0.0005], domain-clustered [-0.0338, +0.0052]),
and it loses on both subsets of the pre-registered split: 0.3566 vs 0.3656 on
the 241 condition-lost queries and 0.3296 vs 0.3518 on the remaining 278. The
loss is thus *smaller* where the arm was designed to help (-0.0090 vs -0.0222),
which is consistent with the reading that body conditions are not what the
ranker is missing, but no subset gains, so the pre-registered bar (gain
concentrated on the lost subset) is not met (`results/CONSTRAINT_SUBSET_cx_qv.json`).

### 5.4 Temporal modelling did not help
Ten probes: year/anchor features, a semantic time-consistency score, oracle and
LLM step decomposition, HyDE, temporal hard negatives, a time-aware constraint
rerank, a train-fitted gate, a fixed year-consistency reordering, and a
coverage diagnostic. Exactly one was strictly pre-registered: it gained +0.0039
([-0.0023, +0.0103], domain [+0.0006, +0.0142], P=0.893) against a +0.02 bar.
The year rule fails cleanly: given the query's interval, ordering the frozen
candidates by year consistency lowers nDCG@10 in 7 of 8 anchored domains (macro
0.2980 → 0.2498). This is a fixed rule losing, not an upper bound: the reordered
list is not the best possible one, and the rule itself is what a temporal ranker
would have to beat. In this corpus a document that mentions another period is
often still the right answer.

### 5.5 The loss form is not the bottleneck
Three alternatives suggested by S5.2 - averaging the log-loss over all
positives, an nDCG-weighted permutation loss (LambdaLoss-style, cutoff 10), and
domain-balanced (sqrt) sampling - were trained on the same 942 groups with the
same hyper-parameters and scored on the same 45-query split:

| objective / sampler | best epoch | val domain-macro nDCG@10 |
|---|---|---|
| current (log-sum-exp over positives) | 2 | 0.2865 |
| mean over positives | 2 | 0.2809 |
| nDCG-weighted permutations | 2 | 0.2764 |
| current + sqrt(domain) sampling | 2 | 0.2912 |
| zero-shot reference | - | 0.2730 |

Selecting on domain-macro rather than on the plain mean picks the same epoch in
all four arms, so the official aggregation does not move the checkpoint choice;
what the internal split cannot do is resolve the gaps themselves (+-0.03, S7), so
of these only the one between sqrt sampling and the current objective (+0.0047,
[-0.0164, +0.0285]) is even measurable, and the 0.010 tie band we fixed before seeing any number
keeps the current objective. The one visible effect of sqrt sampling is that it
raises the share of queries with at least one relevant document in the top 10
from 93.3% to 95.6% without moving nDCG: it makes the easy decision easier, not
the hard one right. This matches S5.2 - what the listwise objective buys is "the
single easiest positive", and none of the three replacements asks for *every*
evidence step to be found.

Dev closes this table the same way it closed the others, and it separates two
effects the paragraph above conflates. The incumbent objective (log-sum-exp over
positives) was retrained through the current code path - same 942 groups, two
epochs, each run taking the epoch its own val prefers (the frozen run picked
epoch 1, this one epoch 2) - and scored on dev: **0.3470**, 0.0111 below the
frozen artifact (domain-clustered CI [-0.0265, +0.0057], 132 better / 167 worse).
That gap is the price of re-running the recipe, not of a new objective: the
objective is unchanged, so it is the baseline against which the four arms above
should be read. They stay unmeasured on dev - the val screen separates them by at
most 0.015, which the dev half-width of 0.007-0.014 cannot resolve - and we do
not carry a sign across splits. What dev supports is narrower than "the
replacements lose": the screen ranks arms at 0.2764-0.2912, it cannot resolve a
0.003 difference, and the one arm run here - the incumbent itself, mid-screen at
0.2865 - reproduces 0.0111 below the frozen number. This is the fifth arm to
lose on dev in this phase of work, and it was the incumbent.

### 5.6 Forcing every evidence step into the top ten does not pay

The remedy S5.2 points to is mechanical: if the fused top ten covers only 59.5%
of annotated steps (51.4% for the cross-encoder alone), require every step to
have a positive ranked high. We trained
the step-augmented listwise objective of S3 at two weights and scored the
resulting systems on dev with the frozen fusion:

| system | dev domain-macro | Δ vs frozen | domain-clustered 95% CI | steps in top 10 | questions with 2+ relevant in top 10 |
|---|---|---:|---|---:|---:|
| frozen (no step term) | 0.3581 | - | - | 59.45% | 38.31% |
| L_step, lambda = 0.3 | 0.3442 | -0.0139 | [-0.0348, +0.0044] | 55.58% | 37.95% |
| L_step, lambda = 1.0 | 0.3438 | -0.0143 | [-0.0351, +0.0037] | 56.00% | 37.88% |

The coverage column counts the fused list (`runs/sub_*`); its top ten does not
change when the list is truncated from 1000 to 100 rows (59.45%,
`results/MULTIEVIDENCE_sft_e2bc_sub100.json`). The cross-encoder's own ordering
over the same 100 candidates covers 51.39% before the step term and 48.14% /
45.78% with it (`results/MULTIEVIDENCE_*_depth100.json`), so the degradation is a
property of the ordering, not of the candidate depth.

The term does not merely fail to pay for itself: it *reduces* the very quantity
it optimises. Both arms also lose on the population they target, the two-step
questions (0.4058 to 0.3910 and 0.3889, domain-clustered CI including zero), and
the loss is spread over all partitions rather than concentrated where coverage
was low. Two readings are consistent with the numbers: a step's positives are
mostly documents the ordinary listwise term already covers (3.44 relevant
documents over 2.34 steps, so a step rarely adds a *different* document), and the
log-sum-exp inside a step is dominated by the step whose positive is hardest to
retrieve, so the gradient chases documents that are absent or buried in the
candidate list rather than assembling the answer. The annotations are not at
fault: every one of the 2,873 step-level positives is also a query-level
positive.

The instructive part is the selection behaviour. On the 45-query internal split
lambda = 1.0 was the *best* of all six arms trained that day (0.2913, above the
current objective's 0.2865), and dev puts it last. At this split size the
internal ranking is not a weak signal but the wrong one, which is why S3 fixes
the fusion form in advance and S7 refuses to select anything on 45 questions.
We report the failure rather than tuning lambda further: the pre-registered bar
was +0.005, and a second sweep of weights on 45 queries would buy noise.

### 5.7 Two further corrections that also lose

The frozen system feeds the reranker a title-only query and excludes the 140
training queries whose dense top-100 contains no positive. Both are defensible
complaints, and both were corrected with the frozen recipe (two epochs, same
schedule) and scored on dev with the frozen fusion:

| system | dev domain-macro | Delta vs frozen | query-clustered 95% CI | domain-clustered 95% CI |
|---|---:|---:|---|---|
| frozen | 0.3581 | - | - | - |
| full-question query, 64-word cap | 0.3443 | -0.0138 | [-0.0272, -0.0005] | [-0.0338, +0.0052] |
| + 140 backfilled queries, 294 positives | 0.3443 | -0.0138 | [-0.0288, +0.0010] | [-0.0311, +0.0012] |

The two corrections cost exactly the same 0.0138, which suggests the frozen
recipe sits in a flat basin in both directions rather than that either
correction is intrinsically bad. The mechanisms differ. For the query arm the
loss is smaller on the subset where the body conditions live (S5.3), so the
extra text does carry signal, but not enough to pay for the harm it does on the
other 278 queries; the reranker is simply better at its job with a short, clean
query. For the backfill arm the mechanism is sharper: the 294 restored positives
are, by construction, documents the first stage never retrieves. Training the
cross-encoder to rank them high therefore spends capacity on a decision it never
gets to make at inference, where the candidate list is the dense top-100. The
correction would only pay if the first stage were fixed too, which the ten
retrieval-side probes of S5.4 did not achieve.

The selection behaviour repeated itself: the full-question arm's 0.2920 was the
highest score of the seven arms trained on 11-10 on the 45-query internal split,
and it lands at 0.3443 on dev, 0.0138 below the system it was meant to improve.
Most of that 0.0138 is not the correction, though. The unchanged objective
re-run through the current code path costs 0.0111 (S5.5), and measured against
that same-code baseline the two corrections here cost 0.0026 each, with
domain-clustered intervals [-0.0187, +0.0100] and [-0.0162, +0.0101] that
include zero; the two step arms above behave the same way (-0.0027 [-0.0154,
+0.0082] and -0.0031 [-0.0153, +0.0072], `results/ATTRIBUTION_LXPOSSUM.json`).
What dev supports is a deployment decision - five of the ladder's six steps were
retrained and none beats the frozen 0.3581, and the sixth (the dense encoder's
pooling) is a wash at +0.0014 - plus a negative result relative to a same-code
baseline that the 45-query screen cannot rank at all: the four arms it scored
on the shared split (0.2865-0.2920) and dev reverses that order.

## 6 Reproducibility and efficiency

Every artefact is content-addressed: document embeddings, dense runs, CE runs and
submissions carry a fingerprint sidecar (`.fp.json`) that binds model, pooling,
max length, query file and fusion parameters, and the submission writer refuses
to reuse a stale cache. The submission path was replayed end to end on synthetic
test splits and passes the organisers' `format_checker.py` with zero errors on
three domains. Seed ensembling of the cross-encoder was tried and rejected: z-averaging the
three seeds before fusion gives 0.3567, i.e. -0.0014 against the single
checkpoint (domain-clustered CI [-0.0032, +0.0001], 481 of 519 topics tied),
so the submission ships one model (`results/ENSEMBLE.md`). Two CE epochs take
55 minutes and dev re-ranking costs ≈34 GPU-minutes for 519 queries, both on one
8 GB laptop GPU; the dense document embeddings are the only large artefact
(several GB on disk).

## 7 Limitations

Domain-macro lets 3-query domains weigh as much as 240-query ones; the temporal
findings are about *textual* time only, not about systems with timestamps;
supervised results are single-corpus and single-family (one dense model, one
cross-encoder). The window/pooling schedule (250-token windows, stride 192, max
over windows, 256-token query cap) was fixed a priori and is not ablated here; the
dense encoder's pooling was ablated late and is a wash (CLS 0.2956 against 0.2943
for mean); the
1:1 fusion weight and the two-epoch schedule were picked on the training-internal
split, so the frozen number carries that selection. The step-supervision results
are reported at one seed; a second seed, and distilling a larger re-ranker into
the 568M encoder (Hinton et al., 2015), were not affordable within the GPU budget
of this submission.

Selection is noisy by construction: the 45-query internal split improves the
sum-positive arm by +0.0135 with a domain-clustered interval of [-0.0317,
+0.0571], i.e. it cannot resolve differences below about 0.04. Every arm here is
therefore screened internally and judged on dev with a paired domain bootstrap
(half-width 0.007-0.014); dev gains below +0.010 are treated as unresolved. The
screen's record is not that it promoted every loser: one arm was its top score
(the full-question arm, 0.2920), the loss sweep's winner fell inside the
pre-registered 0.010 tie band and the base was kept (`LOSS_WINNER.json`), stage 3
kept the same base (`STAGE3_WINNER.json`), one arm was trained to fix a documented
defect rather than because it won, and the step arms were run as a pair. It is
rather that the screen cannot rank them: the four arms it scored on the shared
split lie within 0.006 of each other and dev puts them 0.0026-0.0031 apart from
the same-code baseline, all with intervals that include zero.

Neither is retraining free: the frozen recipe re-run under three seeds gives
0.3575 / 0.3528 / 0.3527 on dev with freshly sampled pools (14.6-14.8k history
documents against 12.5k in the frozen run), and a fourth run through the current
code path 0.3470. The retraining band is therefore 0.347-0.358, and 0.3581 is
the property of the submitted artifact rather than a number the recipe
reproduces on demand; we did not hold negative-pool sampling fixed.

## References

Citations as they will appear in the camera-ready (`\cite{key}` keys in parentheses).

[abdallah2026tempo] Abdelrahman Abdallah, Mohammed Ali, Muhammad Abdul-Mageed, and Adam
Jatowt. 2026. TEMPO: A Realistic Multi-Domain Benchmark for Temporal
Reasoning-Intensive Retrieval. arXiv:2601.09523. (`abdallah2026tempo`)

[abdallah2026reteco] Abdelrahman Abdallah, Mohammed Ali, Muhammad Abdul-Mageed, Kevin Duh, and
Adam Jatowt. 2026. RETECO: A SemEval-2027 Shared Task on Reasoning-Oriented
Retrieval. Task proposal, University of Innsbruck. (`abdallah2026reteco`) The
organizers ask participants to cite TEMPO (and RECOR for Track 2) until the task
description paper appears.

[robertson2009bm25] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic
Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information
Retrieval*, 3(4):333-389. (`robertson2009bm25`)

[nogueira2019passage] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking
with BERT. arXiv:1901.04085. (`nogueira2019passage`)

[cormack2009rrf] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher.
2009. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning
Methods. In *SIGIR 2009*, pages 758-759. (`cormack2009rrf`)

[burges2010lambdarank] Christopher J. C. Burges. 2010. From RankNet to LambdaRank
to LambdaMART: An Overview. Technical Report MSR-TR-2010-82, Microsoft Research.
(`burges2010lambdarank`)

[wang2018lambdaloss] Xuanhui Wang, Cheng Li, Nadav Golbandi, Michael Bendersky, and
Marc Najork. 2018. The LambdaLoss Framework for Ranking Metric Optimization. In
*CIKM 2018*, pages 1313-1322. (`wang2018lambdaloss`)

[jarvelin2002ndcg] Kalervo Jarvelin and Jaana Kekalainen. 2002. Cumulated Gain-based
Evaluation of IR Techniques. *ACM Transactions on Information Systems*,
20(4):422-446. (`jarvelin2002ndcg`)

[hinton2015distilling] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015.
Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
(`hinton2015distilling`)

[chen2024bgem3] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and
Zheng Liu. 2024. BGE M3-Embedding: Multi-Lingual, Multi-Functionality,
Multi-Granularity Text Embeddings Through Self-Knowledge Distillation.
arXiv:2402.03216. (`chen2024bgem3`)

[mxbai2024card] mixedbread.ai. 2024. mxbai-embed-large-v1 model card.
https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1 (`mxbai2024card`) No
peer-reviewed description exists for this checkpoint; we cite the model card and
report the exact revision in `results/MODEL_MANIFEST.json`.

