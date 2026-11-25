# RETECO Track 1a — Temporal Retrieval, SemEval-2027 Task 1

Research artifacts for **Track 1a (Temporal Retrieval)** of the
[RETECO](https://datascienceuibk.github.io/RETECO/) shared task:
given a natural-language query and a domain corpus, return a ranked list of
documents that are relevant **and** consistent with the temporal conditions
implied by the query.

**Current research status (2026-11-20).** The reference submission system
scores **0.3581** on official dev (re-verified: 519 queries, 13 domains, hashes
in `results/FROZEN_BASELINE_20261120.json`). The research question is why a
stronger standalone reranker yields a weaker fused ranking, and whether a
bounded top-ten replacement recovers evidence fusion misses. On the frozen
candidate protocol (at most 3 external candidates against ranks 8-10, at most
one swap per query) a Qwen3-4B judge reaches **0.3646** (+0.0065, 95% CI
[+0.0021, +0.0117]) against a **0.3769** ceiling for that pool; adding retained
context to the same prompt does not help, and off-the-shelf rerankers given the
same pool lose. Details: `results/replacement_v2/REPORT.md`,
`SCORE_COST.md`, `CASES_DEV.md`, `docs/DECISIONS_20261120.md`.

**摘要（中文）**：本仓库是 RETECO Track-1a 的全部实验产物：官方 BM25 基线复现、
bge-base 稠密检索、bge-reranker-v2-m3 交叉编码器精排、z 归一化两视图融合（历史章节保留 RRF 对照）、跨 13 个领域
的 dev 评测，以及时间表示（时间特征、语义时间一致性、查询分解、HyDE）的一系列负结果。
所有数字都由本仓库中的逐查询评分 JSON（`results/score_*_dev.json`）、
dev TREC run 文件与官方 qrels 重算得到，可复现；fused 的 1000 深 run 未提交，
可用 `scripts/fuse_scores.py` 从 dense top-1000 + CE 分数重建。

Status after the second review round: [docs/STATUS_20261121.md](docs/STATUS_20261121.md)

**Score-side experiments (2026-11-22).** The frozen system reproduces end-to-end from the
frozen checkpoint through the documented chain (0.3581821 vs recorded 0.3580985, 11/13
domains equal to four decimals). A lambdarank nDCG@10 retraining of the same CE scores
0.3519 (-0.0062), and a fusion-weight sweep selected on train (w=0.75) drops dev to 0.3486
against 0.3582 at the frozen w=1, so neither the loss function nor the fusion weight
explains the remaining gap.

**First-stage embedder probe (2026-11-22).** Swapping mxbai for Qwen3-Embedding-0.6B on three
dev domains raises recall@100 everywhere (+1.5 to +8.6 points) but the frozen CE + fusion score
is mixed: bitcoin 0.2143->0.2199, quant 0.1538->0.1872, law 0.4357->0.4061 (mean +0.0032,
about +0.0007 on the 13-domain macro), so the full-corpus encode was not run.

**Negative-label audit and facet recall (2026-11-23).** Removing the unjudged hard
negatives from CE training is *worse*, not better: dev 0.3409 when the dense top-10
negatives are dropped and 0.3308 when all hard negatives are dropped, against
0.3582 for the frozen recipe — the unjudged documents are where the cross-encoder
learns the head of the list, even though a model sample says 33%
of them look relevant. Step-level (facet) BM25 recovers a judged document for
113 of the 234 queries that whole-query BM25 misses, but at pool level the
union only lifts zero-query coverage 86 -> 94 and the frozen
cross-encoder converts 5 of the 15 affected queries, i.e. about +0.0015
domain-macro. Details in `results/replacement_v2/REPORT.md` §12.

**Transfer check (2026-11-23).** The widened configuration rerun on the 969 training queries
gives +0.0045 pooled (head-first six pairs), against +0.0154 on dev; the dev split is favourable and
the expected gain on unseen queries is about +0.005. An expected-gain selector fitted on the training
folds scores +0.0042 in grouped CV and +0.0086 on dev, so it does not beat the simple rule.

**Best dev configuration (2026-11-22).** The corrected widened action space judged at six pairs per
query (12 prompts) with head-first ordering reaches **0.3735 (+0.0154, CI [+0.0065,+0.0256])**,
against 0.3646 (+0.0065) for the tail arm at 18 prompts; with the dense+CE prior ordering it is
0.3703 (+0.0122, CI [+0.0054,+0.0203]). The paired advantage over the tail arm includes zero
(+0.0056 [−0.0009,+0.0127]), so the honest claim is a stronger and cheaper configuration, not a
proven replacement for the tail arm. Configuration frozen in
`configs/competition_config_20261122.json`.

**Temporal-misordering ceiling (2026-11-24).** If every *verified* period/direction error above a gold document in the out-of-fold rankings were corrected, the official domain-macro gain would be **+0.0026** (95% CI [+0.0003,+0.0053]) on a stratified 138-query / 533-pair audit — below what the existing six-pair judge already gains out of fold (+0.0045). The automatic screen says +0.0075 but only 5 of its 15 flags survive reading (the rest are page chrome, topical differences, or cases where the document above the gold is better). The audit's larger finding: **14.1% of the documents ranked above gold are flagged as boilerplate/fragments** (a heterogeneous flag, not page chrome: no flagged document contains a navigation phrase; they are truncated fragments, question copies and ordinary answers). A text-only detector for the deployable version gives **+0.0044 fold-honest** ([+0.0004,+0.0092]) at `fused_z − 0.25·z(chrome)`, versus +0.0161 for the oracle demotion of the flag itself. Artefacts `results/analysis/temporal_ceiling_summary.json`, `results/replacement_v2/REPORT.md` section 18.

**Three further nulls (2026-11-24).** Generative reverse scoring (UPR-style, Qwen3-4B, 969
out-of-fold queries, 96,900 pairs) is orthogonal to the fused score (mean per-query Spearman
−0.0048) but far weaker: alone 0.062374, combined −0.000886 against 0.278869. Cheap within-query
spread features cannot predict which queries fail (AUC 0.44–0.48). The text-only boilerplate
detector that gains +0.0044 out of fold does not transfer to dev (0.358108 → 0.357754 at λ=0.25,
λ=0.1 → 0.357777). The acceptance margin is irrelevant below 0.30 and a second or third swap adds
only +0.0003 (dev 0.0154 → 0.0157), so the frozen one-swap 0.3735 configuration stands. Artefact
`results/replacement_v2/threshold_transfer.json`, `results/analysis/upr_abc_train.json`.

**Equal-budget position and replacement audit (2026-11-24).** The six-pair external
policy stays at **0.3735**. Internal exchanges score 0.3676 and a three-plus-three
mixture scores 0.3654 on dev. A fixed three-seed CE ensemble with freshly judged
external actions reaches 0.3658. The [experiment report](docs/POSITION_REPAIR_20261124.md)
traces the missed opportunities through candidate, comparison, and acceptance stages.

**Shared-reference follow-up (2026-11-24).** Scoring twenty documents with the same
Qwen judge and a shared anchor gives **0.3682** on official dev at 38 prompts per
query. Using it only when the six-pair judge abstains gives **0.3741** at 31.3
prompts per query, a paired gain of +0.0006 [−0.0079,+0.0086]. The fallback adds
thirteen net relevant documents to the top ten, with gains at ranks 4–10 offset
by losses at ranks 1–3. The 12-prompt **0.3735** configuration remains the
competition system. See the [same-query comparison and rank accounting](docs/ANCHOR_RERANKER_20261124.md).

**Top-three confirmation (2026-11-24).** A follow-up on the same official-dev queries
scores **0.37554** for fixed protection of the first three ranks and **0.37526**
when a fresh two-order comparison can approve one exchange. Confirmation keeps
19 of 25 helpful and 18 of 23 harmful proposed changes. The score-cost comparison
retains the 12-prompt **0.3735** competition configuration.
[Matched head-confirmation experiment](docs/HEAD_AGREEMENT_20261124.md)

**Reaching zero-score queries (2026-11-24).** With the same six comparisons,
the supervised shortlist raises the 969-query training-fold score by +0.00262
and official dev scores **0.37154** versus **0.37350**. In the direct dev
comparison it recovers four competition zeroes and creates four new zeroes.
The score gain on competition zeroes (+0.00318) is offset by a larger loss on
other queries (−0.00515). [Shortlist and routing analysis](docs/ABSTENTION_SHORTLIST_20261124.md)

**Evidence gained and lost (2026-11-24).** On the same 328 training-fold
queries, Direct scores **0.26573**, four-outcome support **0.25869** and
isolated relative support **0.26194**. A qrels-based veto on the previous
official-dev run would raise **0.37154** to **0.37356** while preserving all
36 helpful swaps. The 12-prompt **0.37350** competition reference remains in use.
[Matched prompts and evidence-step audit](docs/JOINT_SWAP_SUPPORT_20261124.md)

**Rejected-pair evidence audit (2026-11-24).** The actual excerpts for
20 rejected positive pairs already contain useful evidence in seven cases
and cut off useful evidence in five. On all 328 f0 queries, longer
excerpts move **0.26573 → 0.26582**, despite three more accepted
beneficial pairs. [Case review and controlled re-scoring](docs/REJECTED_SWAP_EVIDENCE_20261124.md)

**Matched judge checkpoints.** On the same 104 labeled f0 pairs,
Qwen3-8B accepts **4/27** beneficial and **2/77** harmful pairs; the
original 4B Direct judge accepts **7/27** and **4/77**. Their pair-level
AUCs are both about **0.71**. The specialist Qwen3-Reranker-4B accepts
**12/27** and **16/77** under its pointwise protocol.
[Checkpoint comparison](docs/SWAP_JUDGE_CHECKPOINT_20261124.md)

**Extractive evidence and edit budget (2026-11-25).** Selected and first-span
pointers accept identical pairs even though the inputs differ on 62 of 104
comparisons. Starting from the **0.37350** competition result, a dense top-100
oracle reaches **0.58504** with one fixed-position addition, or **0.69850**
with one addition and free reordering. The protected 20-document pool tops
out at **0.59250**. [Evidence pilot and edit ceilings](docs/SWAP_EVIDENCE_SPANS_20261125.md)

**Matched admission and ranking (2026-11-25).** On the same 65 held-out
queries and dense top-100 scores, starting at the six-pair result, admitting one
document in place gives **0.27723**. Sorting the same ten after admission gives
**0.28642**, and unrestricted ranking gives **0.28703**. On official dev, editing
only six-pair abstentions with the protected 20-document pool gives **0.37549**
for one fixed admission and **0.37453** for the same admission with reordering.
On the two matched held-out cohorts, Qwen3-8B lifts unrestricted top-100 fusion
from **0.28703 to 0.31986**. Its gain with a fixed-slot single swap is **+0.00250**.
[Matched protocols and score breakdown](docs/ANCHOR_EDIT_SCOPE_20261125.md)


**Temporal query contrast (2026-11-25).** In 11 fixed evidence pairs from five
training domains, the zero-shot and task-fine-tuned BGE checkpoints follow
both sides of the changed time condition in 2 and 1 cases. The default Qwen3
0.6B reranker flips correctly in 2 cases, and its event-time instruction in 1.
[Evidence offsets and paired scores](docs/TEMPORAL_QUERY_CONTRAST_20261125.md)

**Full official-dev Qwen3-8B evaluation (2026-11-25).** With the frozen
70:30 blend and 100 dense candidates for each of the 519 official dev queries,
Qwen3-8B scores **0.37959**, compared with **0.37350** for the six-pair
reference (paired difference **+0.00609**, 95% interval [−0.00803, +0.01987]).
All thirteen domains improve against the original frozen system (0.35810).
A secondary replay uses 8B only on the 263 queries where the six-pair method
abstains and reaches **0.38206**. In the matched edit-scope replay, reordering
the same ten scores 0.37994. The top-ten audit attributes the largest positive
component to relevant documents kept in both rankings.
[Official paired results and reproduction](docs/ANCHOR_OFFICIAL_DEV_20261125.md)

**Edit scope and fixed-budget shortlist follow-up (2026-11-25).** A secondary
replay reorders the accepted top tens and applies full 8B on abstentions,
scoring **0.38314** on official dev. Candidate availability at a fixed budget
of six challengers favors searching within the dense top 100 under a single
rank-ten replacement. Across 519 queries, a fixed position-ten 4B replacement
has a lower point estimate for each of the four tested shortlists.
[Matched controls and query-level results](docs/ROUTED_SHORTLISTS_20261125.md)

**Budgeted 8B replay (2026-11-25).** On official dev, adding the highest 24
unlisted dense candidates on six-pair abstentions gives **0.38263** at **45.45**
estimated prompts per query, versus **0.38206** at **112.34** for the full
100-document fallback. Reordering accepted lists with full-pool frozen-score
normalization and only ten 8B scores gives **0.38374** at **54.43** prompts.
[Training-holdout selection and matched normalization controls](docs/BUDGETED_ANCHOR_FALLBACK_20261125.md)

## Replacement study (dev, 2026-11-20)

Everything below uses the frozen system (`0.3581`), the frozen candidate
protocol in `configs/replacement_study_20261120.json`, threshold 0, and at most
one swap per query. Candidates are the union of dense top-20 and full-CE top-20
minus the fused top-10, capped at 3 and paired only against ranks 8-10.

* **Ceiling.** Oracle over exactly this scope: **0.3769** (+0.0188), with a
  beneficial pair on 68/519 queries. The older wider-pool 0.4098 is not this
  ceiling.
* **Direct** (Qwen3-4B-Instruct-2507 judges query + two passages, both orders,
  abstention): **0.3646**, +0.0065, 95% CI [+0.0021, +0.0117]; 32 helpful /
  19 harmful / 273 neutral / 195 no-swap; 9,342 prompts, 5.31M tokens, 873 s,
  11.2 GB.
* **Context** (adds clips of the other retained results): 0.3632, +0.0051;
  paired Direct-Context -0.0015 (CI [-0.0051, +0.0016]) at 1.77x tokens and
  1.74x time, so retained context is not a contribution and the project
  continues with Direct.
* **Controls.** Same-model per-document scoring 0.3598 (+0.0017, ns);
  cheap-gap pair selection 0.3626 (+0.0045, 2,630 prompts). Qwen3-Reranker-4B
  under the model card's official suffix (`<think>...</think>` kept outside the
  truncated passage, left padding, the same 112-word query-matched window as
  Direct): single swap 0.3616 (+0.0035, CI [-0.0032, +0.0105]), full rerank of
  the same thirteen documents 0.3173 (-0.0408, CI [-0.0628, -0.0188]). The
  interval for single swap includes zero, so the honest claim is that the
  off-the-shelf reranker is not better, not that it is broken: an earlier
  right-padded implementation and a control that read document prefixes had
  made it look far worse.
* **FollowIR p-MRR (official metric).** 20/32/52 topics, computed with the mteb
  formula (negative = documents that should drop under the changed instruction
  stay high): Core17 -0.0260 (dense) / -0.0368 (CE) / -0.0433 (frozen);
  News21 -0.0094 / -0.0236 / -0.0208; Robust04 -0.0449 / -0.0214 / -0.0347.
  No cell has a single promoted document, because the instruction pair only
  removes relevance. The nDCG gains above therefore measure ranking quality on
  the changed query text, not instruction-following.
* **Where the recoverable gain is.** Over 519 dev queries there are 4,524
  non-relevant top-10 slots; 2,497 of them have a relevant document inside the
  dense top-100 outside the top-10, and every query can be improved by one
  replacement. Split by position, the recoverable nDCG@10 gain is 43.6% in
  ranks 1-3, 35.0% in 4-7 and only 21.4% in the 8-10 window the frozen protocol
  can touch - the quantitative reason the capped protocol stops at 0.3769.
* **Widening the action space pays, once the inference path is fixed.** Candidates from
  the union of the dense and full-query CE top-100, any of the ten slots, ordered by the
  dense+CE score gap: 0.36634 / 0.37025 / 0.36772 / 0.36934 at 2 / 6 / 12 / 18 judged
  actions (deltas +0.0082 / +0.0122 / +0.0096 / +0.0112, every CI below 0), against
  +0.0065 for the bounded protocol at 18 prompts. At k=6 the stage costs 12 prompts per
  query - two thirds of the bounded arm - and roughly doubles its gain. The same pool on
  the dense-only start gives +0.0108 / +0.0232 / +0.0203 / +0.0170, against +0.0106 for
  the bounded protocol on that start, so four prompts already match the old arm and
  twelve prompts more than double it.
  *Correction (2026-11-22).* The earlier version of this bullet reported the opposite
  (-0.0009 / -0.0016 / -0.0054 / -0.0057). That run tokenised prompts without the chat
  template the frozen arm uses and stored the reversed-order challenger probability
  without swapping the option names, so the "both orders agree" rule was comparing the
  challenger against the incumbent. Re-running the corrected harness reproduces the
  frozen arm on the 1,686 identical comparisons (mean |delta p| 0.0130, 92.8% within
  0.05, 98.9% same side, 415 vs 416 both-order agreements); the numbers above come from
  the same queries, candidates, slots and rule.
* **Query reasoning is neutral at this scale.** Qwen3-4B wrote an explicit analysis for
  every dev query (entities, time/version conditions, exclusions, evidence type,
  keywords); with the frozen dense encoder the domain-macro nDCG@10 is 0.2933 for the
  original query, 0.2930 with the analysis appended, 0.2546 for the analysis alone and
  0.2915 for the short query alone. The published query-reasoning gains use GPT-4o
  analyses with ReasonIR embeddings; with a 4B analyst and this retriever the
  transformation adds nothing.
* **Absolute condition checks do not work.** Asking the judge a yes/no question about
  whether a document satisfies every explicit query condition (13 prompts/query, 6,747
  in total) collapses onto relevance: candidates outside the top-10 are rated satisfying
  by ~0%, so only 3 of 519 queries swap (all neutral) and the macro stays at 0.3581.
  The A/B framing is what makes the judge useful - it forces a comparison with the
  incumbent instead of an absolute verdict.
* **The pool matters; the ordering inside it does not.** Replaying every policy over the
  corrected 18-action cache (that run judged every candidate it selected): gap ordering
  +0.0082 / +0.0122 / +0.0096 / +0.0112, uncertainty ordering -0.0005 / +0.0036 /
  +0.0064 / +0.0112, random *within the judged set* +0.0023 / +0.0132 / +0.0107 /
  +0.0089 at k = 2/6/12/18. A random *subset of the whole action space* - which changes
  which pairs are asked, not just their order - gives +0.0011 / -0.0017 / -0.0058 /
  -0.0036. The gain therefore comes from restricting comparisons to high-gap tail slots;
  once that pool is fixed, which of its pairs is asked first barely matters. Ordering the
  same judgements by the true delta (they still have to pass the judge's rule) buys
  +0.0258 at k=2 and +0.0212 at k=6, so a better prior over *which* pairs to ask is where
  the remaining headroom is.
* **A condition-focused comparison is precise but rare.** Keeping the pool, pairs, orders
  and rule and asking instead which passage better satisfies the conditions the query
  states (time/version, entity/jurisdiction, numeric bounds, exclusions) yields 15 swaps
  (12 helpful and 3 harmful among the 15 swaps that move nDCG, out of 106 swaps made;
  the relevance question moves nDCG in 51 of its 324 swaps) and +0.0044
  (CI [+0.0012, +0.0085]) at the same 9,342 prompts; on the dense start +0.0071
  (CI [+0.0024,+0.0123], 29/10 swaps) and on the reranker start +0.0034
  (CI [-0.0003,+0.0074], 27/8), so the higher hit rate travels across starts. Merging both
  questions into one prompt does not: 0.3626 (+0.0045, CI [-0.0005,+0.0102]), 31/17
  swaps - it collapses to relevance behaviour. Applying the condition swap when it fires
  and the relevance swap otherwise reaches 0.3661 (+0.0080, CI [+0.0030, +0.0138]) but the
  paired gain over the relevance arm is +0.0015 (CI [-0.0001, +0.0044]) at twice the
  prompts, so it is not established.
* **The gain lives where the incumbent is doubtful.** Re-running the bounded protocol against
  ranks 1-3 with the same pool, same pairs and same 9,342 prompts gives 0.3539 (-0.0042,
  CI [-0.0214,+0.0137], 23/33 swaps), whereas ranks 8-10 give 0.3646 (+0.0065). The recoverable
  gain sits in the top ranks, but a pairwise judge cannot cash it in.
* **Two further judge variants do not pay.** Evidence-first generation (quote the deciding span,
  then answer) swaps 4 times on a 39-query sample (2/2, -0.0006) at ~4x cost with 7% unparsable
  output; a cascade that sends only Direct-uncertain queries (230/519) to the condition judge
  fires 13 times, all neutral, reproducing Direct at 26.0 prompts/query.
* **A cost-neutral split between the two questions is flat.** 18 prompts split 6 relevance pairs
  + 3 condition pairs gives 0.3647 (+0.0066) with 29 helpful/13 harmful swaps, versus 0.3645
  (+0.0064) with 32/19 when all nine pairs go to the relevance question: paired +0.0001
  (CI [-0.0048,+0.0048]). Precision rises (69% vs 63%), score does not.
* **Combining the two questions does not help.** Requiring both judges to agree gives 0.3612
  (+0.0031, CI [+0.0008,+0.0061]) with 11 helpful/2 harmful swaps; a condition-question veto
  cuts harmful swaps from 19 to 4 but also helpful swaps from 32 to 18, leaving +0.0035.
* **Held-out replication.** The 969 gate-train queries, ranked by cross-encoders that never
  saw them, show 8,518 non-relevant top-10 slots, 5,161 with a relevant document in the
  dense top-100, and gain shares 44.7/34.1/21.2 across ranks 1-3/4-7/8-10 (dev:
  43.6/35.0/21.4). Constraint labels on 100 stratified held-out slots: temporal 50,
  entity scope 38, numeric 5, exclusion 4, comparison 1, none 2.
* **A learned expected-gain model does not transfer.** Ridge on 8,721 leak-free fold pairs
  (AUC 0.66) applied to dev gives 0.3615 (+0.0034, CI [-0.0015,+0.0088]): worse than the
  fixed relevance rule's 0.3646 (+0.0065).
* **The condition question does not transfer as a constraint filter.** On FollowIR it moves
  the score by 0.0000/+0.0019/+0.0019 (Core17/News21/Robust04) against +0.0084/+0.0072/
  +0.0064 for the relevance question, and removes demoted documents in 0/2, 2/7 and 2/7 of
  its swaps.
* **Budget.** 0 / 1 / 2 / 3 candidates: 0.3581 / 0.3610 / 0.3655 / 0.3646;
  the cap stays at 3 because the cap-2 advantage was observed post hoc.
* **Pair budget (single prediction cache).** Replaying the same Direct
  predictions while judging only k pairs: at k=1 the deltas are -0.0007
  (frozen order), +0.0043 (cheap-gap), +0.0007 (random), +0.0075 (oracle);
  at k=3 +0.0002 / +0.0046 / -0.0014 / +0.0115; at k=9 every ordering returns
  +0.0065. Judging some pair early captures most of the gain. The gap order is
  ahead at k=1 but inside one standard error of a random pair at k=3, so the
  ordering saves calls rather than adding quality. The random ordering uses a
  process-stable hash (`zlib.crc32`) and reproduces bit-exactly.
* **FollowIR transfer (three collections).** Re-running the frozen pipeline on
  Core17, News21 and Robust04 gives graded nDCG@10 (original / changed
  instructions): 0.2674 / 0.1761, 0.3401 / 0.2849, 0.3092 / 0.2382. Fusion
  beats both dense and CE-only in all six instruction splits. Direct then adds
  +0.0084 (CI [-0.0017, +0.0186], 8/3), +0.0072 (CI [-0.0014, +0.0161],
  12/6) and +0.0064 (CI [+0.0018, +0.0114], 13/3): positive everywhere, only
  Robust04 significant.
* **Second starting system.** From the dense ranking (0.2943) the same
  candidates, judge and one-swap rule reach 0.3049 (+0.0106, CI
  [+0.0048, +0.0167]; 61 helpful / 15 harmful), so the mechanism is not
  specific to the fused start and gains more where the top-10 has more
  clearly wrong slots.
* **Qwen3-Reranker-4B starting system.** Reranking the dense top-20 with the
  off-the-shelf reranker gives a 0.2565 start; the same candidates, judge and
  one-swap rule reach 0.2696 (+0.0131, CI [+0.0075, +0.0190]; 57 helpful / 14 harmful).
* **Ablations.** Full 0.3646; without abstention 0.3647; forward order only
  0.3596 (bidirectional agreement is the load-bearing part); without either
  0.3623.
* **Gate.** A logistic-regression gate trained on 969 held-out CE folds
  reproduces Direct (tau 0.0549) but that threshold is not a filter: it equals
  the smallest eligible calibration probability, and its calibration score
  (-0.0014) is worse than doing nothing (0). Re-selecting tau on the
  calibration split alone (0.1026) gives +0.00015 on calibration and +0.00086
  on dev, so the learned gate adds nothing to Direct. The cheap "cross-encoder
  or dense gap" ordering rule is the strongest dev variant at +0.0072; a
  pre-call cheap gate costs 84% of the gain for 15.6% of the judge calls. Fold
  deltas are +0.0007 / +0.0032 / +0.0068 and the 242 calibration queries
  -0.0014: the gain varies with the query split and the starting system, and we
  report the measured dev gain and the other splits rather than projecting a
  fresh-split expectation.

Tables: `results/replacement_v2/SCORE_COST.md`, `SUMMARY_DEV.md`, `REPORT.md`;
per-policy JSONs and TREC runs under `results/replacement_v2/`.

- **p-MRR for the judged arms (2026-11-22).** Rebuilt from the frozen top-ten and the cached
  swap, keyed by domain and instruction condition: Core17 -0.0260 dense, -0.0368 CE, -0.0433
  frozen, **-0.0055 Direct, -0.0051 condition**; News21 -0.0094/-0.0236/-0.0208/-0.0139/-0.0111;
  Robust04 -0.0449/-0.0214/-0.0347/-0.0121/-0.0105. Every system still ignores part of the
  instruction change, but the correction roughly halves the error, which retracts the earlier
  "the new module does not understand instructions" sentence.
- **Capability does not substitute for evidence (2x2 grid, 2026-11-22).** Same candidates, pairs
  and rule: Qwen3-4B at 112 words 0.3646 (+0.0065, 32 helpful / 19 harmful), Qwen3-4B at 400 words
  0.3660 (+0.0079, 36 / 10, 1.55x prompt tokens), Llama-3.1-8B at 112 words 0.3622 (+0.0041,
  24 / 12), Llama-3.1-8B at 400 words 0.3626 (+0.0045, 23 / 7). Paired: +0.0014 for more evidence
  (CI [-0.0008,+0.0044]), -0.0024 for the stronger model (CI [-0.0061,+0.0009]). More evidence
  halves harmful swaps for both judges; a bigger judge does not help at either window.

- **The judges differ in what they touch, not only in how often they fire.** Both arms
  accept about as many swaps (324 vs 330) and both stay positive, but Qwen3-4B changes the
  score with 51 of them against 36 for Llama-3.1-8B, adds a relevant document 31 times
  against 24, and drops a relevant one 19 times against 12. 290 of 519 decisions differ.

- **The labels bound what any of this can claim.** Only 228 of 4,671 candidate pairs (4.9%) have a
  judged challenger, and all 228 are relevant; 92.1% of slot-8 incumbents are unjudged. Within the
  judged subset neither judge is ever demonstrably wrong, and all measured harm comes from
  inserting unjudged documents, so harmful-swap counts are upper bounds, not measurements.
- **Most of the measured harm is a label artefact.** Reading the 19 score-changing swaps
  that remove a judged-relevant document, 8 insert a document at least as useful, 4 are
  about as useful, 3 answer a different aspect and 4 are genuine losses; two of those four
  are navigational pages with no prose. With judgments covering 4.9% of the candidate pool,
  harmful-swap counts are upper bounds, and a document-type filter would remove half of the
  real harm for free.

- **Better-chosen 112 words do not recover the 400-word gain.** With the pool, pairs,
  prompt, model and decision rule fixed, a window chosen by the question's conditions gives
  0.3645 (+0.0064) and two 56-word windows 0.3613 (+0.0032), against 0.3646 for the frozen
  overlap window; paired intervals [-0.0006,+0.0016] and [-0.0038,+0.0020]. Only the amount of
  raw text moves the harmful-swap count (19 to 10 at 400 words), and that interval also
  includes zero, so the capability/evidence programme stops here and the frozen 112-word
  configuration is what we report.


- **Review round 3 (2026-11-24):** nDCG loss with the mask fixed scores 0.3514 against a possum control on the same 969 groups 0.3572 (single-factor, so the loss is not the limit); Diver-GroupRank-7B on the dense top-20 is -0.0011 nDCG@10 over 65 queries but moves +7 relevant documents into the top ten; the 0.3735 configuration loses to accept/reject (26 of 85 queries with a reachable beneficial action reject them all) and no selection rule recovers it; step-level support judgement reaches AUC 0.816 against 0.803 for the global question.
  **Requirement-conditioned evidence (2026-11-24):** on the held-out folds (754 queries) neither selecting passages by the question's requirement (-0.0036 vs the six-pair rule, CI [-0.0074,-0.0003]) nor asking the judge the requirement-conditioned question (-0.0010, ns) helps, and the matched requirement beats a mismatched one by only +0.0010 (CI [-0.0033,+0.0055]). The six-pair configuration stays as the frozen competition entry.
  The pool-100 cascade does not survive more queries either: 0.2453 -> 0.2321 (-0.0133) at 65 queries, so the specialist reranker is nDCG-neutral or worse at every sample size we measured.

## Paper

`paper/main.pdf` is the SemEval-2027 system-paper draft, built from
`docs/PAPER_4PAGE.md` by `scripts/md2tex.py` against the official ACL style:

```
python scripts/md2tex.py docs/PAPER_4PAGE.md paper/main.tex
cd paper && pdflatex main.tex && pdflatex main.tex
```

State: **draft, not submitted**. Body ends inside page 5 (the SemEval review
limit), references pages 5-6, Appendix A (the whole six-step ladder) pages 6-8,
0 overfull boxes. Title follows the required "Team Name at SemEval-2027 Task 1:"
form with placeholder author fields. Headline numbers are checked against their
artefacts by `scripts/check_claims.py` (`results/CLAIMS.md`).
The [November 22 expanded draft](paper/archive/expanded_20261122.pdf) and
its [LaTeX source](paper/archive/expanded_20261122.tex) are archived separately.

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
                    per-query z-norm, score sum (w_dense : w_CE = 1 : 1)
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
* Fusion: per-query z-normalisation of both score lists, then a weighted sum with
  `w_dense : w_CE = 1 : 1` (`scripts/fuse_scores.py --norm z --wa 1 --wb 1`) - the
  form fixed before fine-tuning. Reciprocal rank fusion (`scripts/fuse_rrf.py`,
  `c = 60`) is the earlier release's form and is kept below as a comparison; the
  out-of-fold weight refit selects RRF 1:4, which does not clear the
  pre-registered bar at that form, so the submission stays on z 1:1.

**Reporting convention (corrected 2026-11-09).** The official Track-1a
aggregation is **domain-macro**: the Evaluation Plan says nDCG@10 is
"computed for each query, averaged over the queries of each domain, and then
averaged over the 13 domains, so every domain counts equally", and missing
queries score 0. `iota` (3 dev queries) therefore weighs as much as `history`
(240). `topic-macro` (519-query mean) is reported as a diagnostic only. Both
are computed by `scripts/aggregate_scores.py`; every table states which it uses.

| system, dev (519 queries, 13 domains) | domain-macro (official) | topic-macro |
|---|---:|---:|
| **dense mxbai + SFT CE (2 epochs), z 1:1 (frozen submission)** | **0.3581** | 0.3644 |
| same recipe, fresh checkpoint (reproducibility check), z 1:1 | 0.3583 | 0.3680 |
| same recipe, seed 23, z 1:1 | 0.3575 | 0.3649 |
| same recipe, seed 29, z 1:1 | 0.3528 | 0.3628 |
| same recipe, fresh checkpoint, RRF 1:1 (form fixed for V1) | 0.3431 | 0.3433 |
| same recipe, fresh checkpoint, weight fitted out-of-fold (RRF 1:4) | 0.3187 | 0.3442 |
| dense + zero-shot CE, z 1:1 | 0.3249 | 0.3260 |
| dense + zero-shot CE, RRF 1:1 (V1, first release) | 0.3166 | 0.3199 |
| same recipe, retrained through the current code path, z 1:1 | 0.3470 | 0.3590 |

> **Phase 11 correction (2026-11-10).** An earlier version of this README said the
> 0.3581 headline did not reproduce. That was wrong: the score files for the fresh
> checkpoints were written from intermediate TREC files and never refreshed
> (`results/PHASE11_VALIDATION.md`, "Incident 2"). After re-scoring every fused
> arm from the committed runs, the frozen system is **0.3581** and three
> independent training runs of the same recipe give **0.3583 / 0.3575 / 0.3528**
> (seed spread 0.0055). The fine-tuned reranker is +0.0332 over the zero-shot arm
> at the same z 1:1 form (domain-clustered CI [+0.0192,+0.0472]) and still
> +0.0205 at a fusion weight fitted on 45 held-out training queries with no dev
> signal, so the gain is not a product of the fusion-weight choice. A guard
> (`scripts/check_score_freshness.py`) now fails on score files older than their
> run files.

> **Phase 17 (2026-11-12): five dev-tested variants, all negative.** The loss,
> representation and step-supervision ladders were closed on dev. Every arm that
> the 45-query internal split promoted loses 0.011-0.014 on dev, including the
> incumbent objective retrained through the current code path (0.3470): the
> internal split has essentially no ranking information at that size.
> Step supervision costs 0.0139/0.0143 and *lowers* step coverage from 59.5% to
> 55.6/56.0%; the constraint-faithful 64-word query and the 140 backfilled
> queries cost 0.0138 each. One of them is a "+0.0067 win" under topic-macro and a
> "-0.0138 loss" under the official domain-macro, which is why every table here
> states its convention. The frozen 0.3581 stays the submission; the recipe's
> retraining band is 0.347-0.358 and the frozen artifact sits at its top.
> The remaining ladder step (the dense encoder's pooling) was run as a baseline
> correction: CLS pooling gives 0.2956 against 0.2943 for mean on the 13-domain
> dev set (+0.0014, topic-macro the other way), i.e. a wash - the baseline was not
> held down by the pooling choice.
> **Attribution correction (2026-11-13, after external review).** Most of the
> 0.0138-0.0143 gap is not the variants: re-running the *unchanged* objective
> through the current code path already costs 0.0111 (0.3470 against 0.3581), and
> against that same-code baseline the four arms cost only 0.0026-0.0031, every
> domain-clustered interval including zero (`results/ATTRIBUTION_LXPOSSUM.json`).
> What dev supports is the deployment decision (keep the frozen artifact) and a
> negative result relative to a same-code baseline, not the claim that each edit
> is intrinsically bad.
> **Depth correction.** The 96.3% "steps retrievable" figure is at depth 1000;
> at the 100 candidates the cross-encoder actually sees it is 80.7% (88.3% of
> queries keep a positive), so recall is a partial limit rather than a non-issue
> (`results/MULTIEVIDENCE_sft_e2bc_depth100.json`).
> Details: `docs/RESEARCH_LOG.md` section 5p, `docs/PAPER_4PAGE.md` sections
> 5.2 and 5.5-5.7.

> **Replacement study frozen (2026-11-20).** The 0.3581 arm was re-scored end to end
> from the 13 committed dev run files with the official metric: **0.3580985233**
> domain-macro (topic-macro 0.3643666957), identical to the recorded value, with all
> 519 queries in all 13 domains counted and none missing from the runs. Checkpoint,
> run, qrels, example and evaluation-script hashes are pinned in
> `results/FROZEN_BASELINE_20261120.json`; the study protocol is
> `configs/replacement_study_20261120.json`.
>
> The replacement candidate set is now frozen as well: per query, the union of the
> dense top-20 and the full-query CE top-20 minus the frozen top-10, ordered by best
> rank (ties: dense before CE, then docid), capped at **3** candidates; every
> candidate is compared with frozen ranks 8-10 (9 pairs per query) and a policy may
> execute at most **one** swap. Candidate generation never reads qrels. On dev this
> protocol's oracle ceiling is **0.3769** (+0.0188): 68 of 519 queries contain at
> least one beneficial pair (209 of 4,671 pairs are positive) and 15 of those queries
> currently score zero. The 0.4098 of the earlier research update came from an
> *uncapped* candidate pool and is **not** the ceiling of this protocol. Using the
> title-query CE as the second source instead gives 0.3737, so the source choice is
> documented rather than assumed.

> **Dev replacement results (2026-11-20, frozen candidates, threshold 0, one swap).**
> Direct (Qwen3-4B-Instruct-2507 judges A/B with both presentation orders, abstention
> allowed) reaches **0.36461** (+0.00651, 95% CI [+0.00206, +0.01172]; 32 helpful /
> 19 harmful / 273 neutral / 195 no-swap queries; 9,342 prompts, 5.31M prompt tokens,
> 873 s, 11.2 GB peak). That is 35% of the 0.0188 protocol ceiling. Adding excerpts of
> the nine retained documents (Context) gives 0.36315 (+0.00505, CI [+0.00126,
> +0.00943]), but the paired Direct-minus-Context difference is **-0.00146**
> (CI [-0.00506, +0.00155], p=0.81) while costing 1.77x the tokens and 1.74x the wall
> time, so retained context buys no quality. Same-model per-document scoring gives
> 0.35979 (not significant) and cheap-gap pair selection gives 0.36262 with only
> 2,630 prompts. Artefacts: `results/replacement_v2/`.

Official Track-1a dev BM25 baseline is **0.1147**, i.e. the system is ≈3.1× the
official lexical baseline. TEMPO's best reported system (32.0) is on a different
1,730-query split and is not directly comparable.

**Phase 9–14 update (current system).** The first stage stays
`mxbai-embed-large-v1` top-100; the reranker is `bge-reranker-v2-m3` fine-tuned
on the 1,211 train queries (2 epochs, peak lr 1.5e-5, 8 of 24 layers frozen) and
fused with the dense scores after per-query z-normalisation at **1:1**, a weight
fitted once on train. Dev: **0.3581** domain-macro vs 0.3249 for the zero-shot
fused arm (**+0.0334**, domain-clustered [+0.0191,+0.0477]) and 0.2943 for dense
alone; two further seeds give 0.3575 and 0.3528. Weight refitting out-of-fold
selects RRF 1:4 and keeps +0.0205 [+0.0010,+0.0404] over the zero-shot arm at the
same weight, so the inherited z 1:1 form stays the system. Sections below keep the
Phase-8 history and the older RRF configuration for the record.

**Phase 8 update (historical).** Swapping only the first stage to
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

**Diagnosis.** Inside the frozen top-100 pool the oracle nDCG@10 is **0.7446**
while the frozen system scores 0.3581 and dense alone 0.2943 — the reranker
captures **14.2 %** of that headroom (per domain 0-63 %; 41 % of queries improve,
19 % worsen). The zero-shot cross-encoder's within-query AUC (0.801) is no better
than the dense retriever's (0.806), and ten post-hoc interventions failed. The
binding constraint is the scorer, not recall in the aggregate (a second dense
retriever lifts Recall@100 0.699 -> 0.737 and oracle@10 0.7446 -> 0.7784, worth
about +0.003 at 14 % capture) - but **26.6 % of dev queries score exactly zero
under the frozen system, and 37.7 % of those have no gold document anywhere in
the top-100**, so recall is the binding constraint for that quarter of the
failures (`results/CAPTURE_CURVE.md`, `results/DEV_ERROR_ANALYSIS.md`).

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

**Result under the official metric.** Two schedules were run; the longer one is
the frozen system and the shorter one is the ablation:

| dev (519 topics) | domain-macro | topic-macro |
|---|---:|---:|
| dense + zero-shot CE, z 1:1 | 0.3249 | 0.3260 |
| dense + SFT CE, z 1:1 (1 epoch, lr 1e-5) | 0.3336 | 0.3339 |
| **dense + SFT CE, z 1:1 (2 epochs, lr 1.5e-5)** | **0.3581** | **0.3644** |
| dense + SFT CE, z 1:1 (2 epochs, **seed 29**, replication) | 0.3527 | 0.3624 |

The 2-epoch arm vs zero-shot at the same frozen weight: **+0.0332**,
query-bootstrap CI [+0.0161,+0.0523], domain-clustered [+0.0192,+0.0472],
188 queries better / 78 worse / 253 tied, 11 of 13 domains up. The 1-epoch arm
gives **+0.0087** [+0.0025,+0.0151] / domain-clustered [+0.0037,+0.0142]; it is
the ablation that isolates "more training" from "any training". The total over
V1 for the 1-epoch arm is **+0.0170** [+0.0055,+0.0290], domain-clustered
[+0.0052,+0.0287], which bundles the CE swap with a fusion-form change worth
+0.0083 whose own CI crosses zero — the 2-epoch arm is therefore reported as a
single-factor claim against the zero-shot fusion, not against V1.

**Seed replication.** A second training seed (29, identical recipe and
checkpoint-selection rule) reaches 0.3527: **+0.0278** over the same zero-shot
baseline, query CI [+0.0118,+0.0455], domain-clustered [+0.0159,+0.0404],
181/82/256, and 0.3123 vs 0.2832 on the 12-domain view. Both seeds are positive
with domain CIs excluding zero, and the seed-to-seed spread (0.0054) is far below
the delta, so the honest headline is **+0.031 ± 0.003 over the strongest
zero-shot baseline**; 0.3581 stays the frozen submission number. Details:
`results/REPLICATION.md`.

The pre-registered adoption bar (+0.015 on dev **topic-macro**) was **not** met
by the 1-epoch arm: under the registered statistic SFT − V1 = 0.33393 − 0.31987 =
+0.01406. The bar is not restated under the corrected metric; the significance
claims above are made with the official domain-macro bootstrap instead. Frozen
as `runs/sub_sft_e2bc_<domain>_dev.trec` (13/13 official-checked, de-tied).
See `results/PHASE9_PLAN.md`, `docs/RESEARCH_LOG.md` §5i/§5l.

**The SFT arm inherited a weight tuned on the zero-shot arm.** One global scalar
was fitted once on the 1,211 train queries with the zero-shot CE (1:1), frozen,
and never re-fitted; the SFT arms are therefore the *under-tuned* side, so their
measured gains are conservative rather than inflated. To show that the claim does
not hinge on that choice, `scripts/weight_sweep.py` re-fuses the stored
per-query scores at nine weights on dev, for each arm separately (diagnostic
only — nothing is selected on dev). For the frozen 2-epoch arm:

| w (CE) | zero-shot | SFT (2 epochs) | delta | 95% domain-clustered CI | domains up |
|---:|---:|---:|---:|---|---:|
| 0.5 | 0.3208 | 0.3297 | +0.0089 | [+0.0009,+0.0170] | 9/13 |
| 0.7 | 0.3294 | 0.3472 | +0.0178 | [+0.0102,+0.0261] | 13/13 |
| 0.85 | 0.3261 | 0.3518 | +0.0258 | [+0.0158,+0.0351] | 11/13 |
| **1** | 0.3249 | **0.3581** | **+0.0332** | **[+0.0195,+0.0482]** | 11/13 |
| 1.15 | 0.3271 | 0.3552 | +0.0280 | [+0.0168,+0.0393] | 11/13 |
| 1.3 | 0.3278 | 0.3559 | +0.0281 | [+0.0174,+0.0387] | 11/13 |
| 1.5 | 0.3253 | 0.3592 | +0.0339 | [+0.0230,+0.0451] | 13/13 |
| 1.8 | 0.3192 | 0.3538 | +0.0346 | [+0.0194,+0.0535] | 13/13 |
| 2 | 0.3193 | 0.3481 | +0.0287 | [+0.0118,+0.0498] | 11/13 |

Every weight is positive, and at 0.7/1.5/1.8 all 13 domains improve, so the
result is not a lucky weight. The 1-epoch arm's sweep is kept as the ablation:

| w (CE) | zero-shot | SFT (1 epoch) | delta | 95% domain-clustered CI | domains up |
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

There the surface is flat to ~±0.008 with no monotone trend. We report the
frozen-weight number in both cases, never a dev-best one.
(`results/WEIGHT_SWEEP_E2.md`, `results/weight_sweep_e2_dev.json`,
`results/WEIGHT_SWEEP.md`.)

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
