# Separate temporal contradictions from missing evidence

The frozen competition reference is the four-list consensus and confirmed first-position swap, at 0.40674 on official dev. Its evidence grader gives the same zero rating to irrelevant text and explicit temporal conflict. This experiment asks whether a scorer that identifies the two cases separately improves ranking. The distinction matters when a date is unknown, the passage describes the wrong time, or a later source correctly describes an earlier state. [Frozen reference and previous evidence controls](EVIDENCE_REASONER_20261125.md)

## Protocol fixed before the scoring runs

The new grader uses the existing 4B checkpoint, full question and same two document excerpts. It returns logits for four categories. The categories represent unrelated content, topical content with insufficient evidence, apparent evidence with an explicitly conflicting time or version, and usable evidence for at least one question requirement. It treats the time of the supporting claim as the relevant time. No generated explanation or additional inference call is required for the conflict signal.

Keep the existing candidate IDs at widths 14 and 22. At each width, compare a fifth vote based on the plain usable-evidence log odds with an otherwise identical vote that subtracts the margin of a dominant contradiction category. Both use one set of predictions and the same rank prior. Also compare two single-edit rules, which either demote a confidently conflicting document within the existing ten or replace it with a strongly supported outsider. Include the complete frozen consensus and head rule as a selectable reference.

Evaluate the fixed policies on the existing 186 training holdouts. Report three-fold selection, the 65-query exact-reference subset, and the smallest leave-one-domain-out improvement. Hold official dev results until one choice has been made from training. For a second sample, select training query IDs that are disjoint from the 186 before reading their labels, and reuse out-of-fold first-stage runs. The existing 186 are repeatedly used for diagnosis and selection and therefore give a less independent check than the additional sample.

The matched comparison tests an explicit conflict score within the same evidence, candidates and inference budget. It differs from query time encoding and manual date switches by asking whether a document's own claim contradicts a requested condition. [Temporal counterfactual diagnostic](TEMPORAL_QUERY_CONTRAST_20261125.md) · [Q-TIE](https://aclanthology.org/2026.emnlp-main.2611/) · [TEMPS](https://arxiv.org/abs/2609.28048)

## Initial training results

| 186-query training holdout | Domain-macro nDCG@10 |
| --- | ---: |
| Frozen reference | 0.33821 |
| Fifth vote, 14 documents, direct usable-evidence log odds | 0.33836 |
| Fifth vote, 14 documents, with explicit conflict margin | 0.33984 |
| Fifth vote, 22 documents, both controls | 0.33682 |
| Single replacement for strongest conflict, pooled training choice | 0.34019 |
| Rule selected on the other two folds | 0.33902 |

The matched 14-document conflict margin adds +0.00148 to direct evidence scoring. The fold-wise choice adds +0.00081 [−0.00101, +0.00271] over the reference, with +0.00330 on the 65 exact-reference queries. The fixed replacement changes 15 ranked lists. One improves and fourteen have the same nDCG. The single improved Travel query accounts for the fixed rule's pooled gain. The disjoint confirmation spans 155 training queries across 12 domains. The 186-query sample includes all eligible IOTA training questions. [Held-out rankings, logits and disjoint query IDs](../results/analysis/temporal_conflict_reranking_holdout_20261126.tar.gz)

## Test whether a candidate adds a missing answer

The previous replacement step reacts to a possible conflict in an incumbent. A complementary arm seeks affirmative proof for a part of the question the current list covers poorly. Use up to four literal question fragments ending in a question mark, with the full original question retained as context. Score each document once. The model identifies the best supported numbered fragment, no usable support, or support for a different part of the complete question. Keep all 22 candidate IDs and score any additional retained documents as well.

Allow at most one edit, at positions six through ten, where the candidate's predicted overall relevance exceeds that of the displaced document. In the matched pointwise control, select the edit by its discounted overall relevance gain. The coverage variants add the predicted change in the best available support across question fragments, or also require that change to be positive. Use the same prompts and document scores for all arms. The 165 annotated questions permit a ceiling check. A perfect single tail exchange can raise nDCG while adding an annotated step on 29 of those questions, with an ideal joint nDCG increase of +0.03041 across the annotated domain means. This is a diagnostic from known labels. The selection rules use model outputs at inference.

Of the 186 holdouts, 145 have at least two extracted requests. The pointwise and coverage exchanges both score 0.33787 against 0.33821 for the frozen list. Requiring positive coverage scores 0.33578. Across folds, selection scores 0.33638. [Matched exchanges and facet logits](../results/analysis/temporal_facet_reranking_holdout_20261126.tar.gz)

## Select which documents receive the detailed evidence assessment

Apply the same facet logits one stage earlier. Begin with the ten original documents and use the 22-document pool as the candidate source. Choose two challengers from each of the same two tiers, using either predicted overall relevance or relevance plus predicted improvement in coverage of individual question fragments. Then apply the frozen evidence-first scores and ranking consensus to the resulting fourteen. Both selectors receive 22 inexpensive category predictions and allocate exactly 14 detailed assessments. The holdout replay already has the original and additional detailed scores. For a new sample, reuse the original 14 scores and assess only extra documents required by either matched selector.

| 186-query training holdout | Domain-macro nDCG@10 |
| --- | ---: |
| Frozen candidate selection | 0.33821 |
| Select by category-based relevance | 0.34132 |
| Select by relevance and estimated coverage | 0.34147 |
| Selection on the other two folds | **0.34107** |

The fold-wise gain is **+0.00286** [−0.00192, +0.00808]. The minimum estimated gain after dropping one domain is +0.00051. The matched gain from adding coverage to the selector is +0.00014 [−0.00167, +0.00266]. Most of the improvement comes from selecting candidates with a direct evidence criterion before spending detailed assessment slots. The distinction is relevant to stage placement in a cascade, and the closest prior work includes joint evidence selection and explicit aspect diversification. [SetR](https://aclanthology.org/2025.acl-long.861/) · [Joint Passage Ranking](https://aclanthology.org/2021.emnlp-main.560/) · [Holdout rankings and paired effect](../results/analysis/temporal_facet_reranking_holdout_20261126.tar.gz)

Freeze both training choices for the 155-query confirmation. Its starting candidate lists use query-disjoint first-stage folds. Refit the prior repair on the original 186 and rerun the remaining stages on the confirmation questions. The official-dev comparison is locked to one choice. Eligibility requires a positive confirmation estimate that stays positive after omitting any domain. If both choices qualify, use the one with the larger original cross-fold gain.

## Disjoint confirmation and the official-dev choice

| Fixed ranking rule | Initial 186 training holdouts | Disjoint 155 training queries |
| --- | ---: | ---: |
| Frozen reference | 0.33821 | 0.32381 |
| Single replacement for strongest conflict | 0.34019 | 0.32320 |
| Select candidates by relevance and estimated coverage | 0.34147 | 0.32306 |
| Fifth vote using usable-evidence log odds at width 14 | 0.33836 | 0.32912 |
| Fifth vote with conflict margin at width 14 | 0.33984 | 0.32926 |

The original replacement and shortlist rules change the disjoint-set score by −0.00061 and −0.00075. Neither clears the confirmation rule. The shortlist replay obtains the grades for 303 additional documents out of a possible 1,240 by checking only the documents selected by one of its matched candidate screens.

Before any official dev scoring in this round, select the previously specified 14-document vote with a conflict margin as the fallback. Its fixed-rule changes are +0.00163 and +0.00546 on the two training cohorts. The minimum leave-one-domain-out changes are +0.00016 and +0.00339. This additional selection uses both cohorts as development evidence and reserves official dev for evaluating one fixed policy. The contrast against the matching plain evidence vote is +0.00148 and +0.00014 on those cohorts. [Independent confirmation rankings and selection records](../results/analysis/temporal_conflict_disjoint_confirmation_20261126.tar.gz)

The locked vote scores **0.40614** on 519 official dev queries, a change of −0.00060 [−0.00459, +0.00329] from 0.40674. Its smallest leave-one-domain-out change is −0.00203. The plain matched vote scores 0.40613. Explicit conflict scoring adds +0.00001 to that control. The standalone replacement control reports 0.40717 on dev and had failed disjoint confirmation, so the research and submission reference remains the frozen 0.40674 configuration. [Official dev scores, category logits and locked choice](../results/analysis/temporal_conflict_official_20261126.tar.gz)

## Learn the gain of changing one document decision

Compare a document relevance model with a three-class pair model of a useful, neutral or harmful edit. Both receive the same two reranker scores, evidence grade and verification indicators, conflict and support logits, and four individual ranking signals. Both enumerate one replacement below rank one or one within-list swap involving a position in the first five. The learned pair model receives matched forward and reverse examples. Training weights emphasize the change in rank discount and balance domains. At inference, compare choosing the best positive expected gain with requiring a minimum predicted gain of 0.02.

Train on the original 186 and score the 155 confirmation queries. Train again on the 155 and score the original 186. All four learned rules go down in both directions. The guarded pointwise model loses 0.00109 across the 341 disjoint training examples, and the guarded pair model loses 0.00201. This direct objective comparison shows that learning the sign of a single edit does not transfer well with the current evidence scores and populations. [Disjoint out-of-sample edit replay](../results/analysis/temporal_expected_edit_gain_20261126.tar.gz)

## Reproduce the ranking comparisons

Start from the 22-document manifests in the [evidence-reasoner protocol](EVIDENCE_REASONER_20261125.md#replay). Run `eval_temporal_evidence_categories.py score` with the frozen 4B-Instruct model and the corresponding manifest. Replay with `eval_temporal_evidence_categories.py evaluate`, passing the original consensus report, first-position report and 14-document evidence score shards. Pass the locked rule record as `--selection` for official dev. Use `--confirmation` together with the original selection record to reproduce the disjoint train replay.

Use `score --facets` for the question-fragment logits. The tail exchange uses `replay_evidence_facet_exchange.py`. Candidate selection uses `replay_facet_shortlist.py`, the same facet logits and detailed scores for the candidates used by each selector. The original 14 scores reproduce the fixed reference exactly. The extra scoring pool can be obtained from `extra_grading_pool` before reading judgments.

Rebuild the disjoint populations by filtering the complete training candidate manifest to the archived query IDs. Use the same 30-to-60 expansion and checkpoint instructions as the [width study](QWEN_ADAPTATION_AND_PAIR_EVIDENCE_20261125.md#replay). The shipped `build_temporal_confirmation.py` rebuilds the prior repair from the 186 training questions and assembles the independent-set consensus after pointwise and evidence scoring.

## Retrieve local evidence without discarding the existing candidates

Keep the first 30 candidates, use the mxbai top 300 plus the current pool as the shared source, and reselect the other 30. For each document, encode up to three windows with the original mxbai checkpoint. The full-question score takes the best matching window. The clause score also uses literal interrogative fragments of the original question, with per-query normalization of the fragment scores. The source list and candidate budget are shared across all four controls.

| Candidate source | Initial 186 ideal nDCG@10 | Disjoint 155 ideal nDCG@10 | Initial / disjoint queries without a positive |
| --- | ---: | ---: | ---: |
| Existing 60 | 0.67516 | 0.66427 | 31 / 31 |
| Dense ranks | 0.68243 | 0.65382 | 30 / 32 |
| Full question matched to windows | 0.68363 | 0.66426 | 30 / 31 |
| Question fragments matched to windows | 0.69077 | 0.66219 | 28 / 32 |

Replacing half the pool with fragment-selected documents raises the oracle by 0.01562 [−0.00605, +0.03652] on the initial sample. The same rule loses 0.00208 in disjoint confirmation. Each result diagnoses the candidate lists using official training judgments.

In a second setup, retain all 60 existing candidates and score only four new documents from each alternative source. The fragment source finds a new judged positive on eight queries in each population. Plain dense finds one on five and six, respectively. Check the same literal quotation, expected grade and minimum 0.5 grade advantage for both sources. Also require the new candidate to score higher under the 8B pointwise model. Compare allowing one edit at positions 6–10 or 3–10.

| Admission policy | Initial 186 nDCG@10 | Disjoint 155 nDCG@10 |
| --- | ---: | ---: |
| Frozen ranking | 0.33821 | 0.32381 |
| Four new dense candidates, positions 3–10 | 0.33891 | 0.32556 |
| Four new fragment candidates, positions 3–10 | 0.33984 | 0.32547 |

The fixed fragment rule gains 0.00163 on the initial sample, with two helpful changes and no harmful changes among 34 altered lists. The fold-wise selection ties the frozen ranking. Disjoint confirmation gains 0.00166 [−0.00481, +0.01106], with one helpful and two harmful changes among 29 altered lists. The smallest leave-one-domain-out change is −0.00262. The matched fragment-minus-dense contrast in confirmation is −0.00009. Leave this variant at the training stage. [Candidate pools, logits, quotations' verification indicators and paired rankings](../results/analysis/passage-and-conditional-replay-20261126.tar.gz)

## Compare two ways to score the temporal condition itself

The paired prompt protocol applies temporal requirements in one score and asks the same 4B judge to set them aside in the other. Keep the explicit query, excerpts, 14 candidate IDs and checkpoint identical. Add an equal fifth rank vote, using the full score, the relaxed score, the strict-minus-relaxed difference, or an average of the normalized strict score and the difference. The 186-query reference is 0.33821. The four fixed controls score 0.33760, 0.33790, 0.33595 and 0.33631. Fold-wise selection scores 0.33529.

The question-likelihood protocol instead scores how readily the same model would produce the original question after reading the fixed excerpts. A single causal language model pass yields the average target-token log likelihood over the whole question, literal interrogatives and explicitly temporal terms. Compare averaging all tokens with combining the strongest normalized fragment score with the whole-question score. Then give one-quarter weight to the temporal tokens, using the full question score when they are absent. The three matched controls score 0.33556, 0.33428 and 0.33530 as an additional equal rank vote. Each fold selects the reference. These controls test the contribution of the score construction at equal evidence budgets. [Zero-shot question-generation reranking](https://arxiv.org/abs/2204.07496) · [Cached likelihoods and full replay](../results/analysis/passage-and-conditional-replay-20261126.tar.gz)

## Reserve training queries for future confirmation

A third sample contains 191 official training questions in eight domains. Select up to 30 from each remaining domain by a fixed SHA-256 ordering of its domain and query ID after excluding the preceding 341 questions. Five domains have already supplied every eligible question to the preceding studies. This reserve is disjoint from the two samples used to compare these changes. Previous project components may have encountered these questions in other training or analyses. Freeze a new ranking policy before using these 191 relevance judgments. Keep the 519 public-dev queries out of policy selection. [Reserve IDs and sample definition](../results/analysis/passage-and-conditional-replay-20261126.tar.gz)

## Change presentation order before selecting the first result

Compare the same three leading documents of the frozen system using Qwen3-4B-Instruct-2507. Keep the full query and two fixed excerpts per document. The model can select a document or indicate that none offers usable evidence. Give each document each presentation position twice, using all six permutations. The single-order control uses the reference order. The balanced rule averages the aligned logits. The guarded rule requires a strict majority for the proposed replacement and a mean logit gap of at least log(2) over the incumbent. Every rule can exchange only rank one with rank two or three.

| Selection policy | Initial 186 nDCG@10 | Disjoint 155 nDCG@10 |
| --- | ---: | ---: |
| Frozen ranking | 0.33821 | 0.32381 |
| Single presentation in reference order | 0.32559 | 0.31813 |
| Average over six presentations | 0.33876 | 0.32193 |
| Majority and margin guard | 0.33930 | 0.31631 |

Balancing presentations gains 0.01317 over the single-order arm on the initial 186 and 0.00379 on the disjoint 155. Choosing the rule on other folds gives 0.33475 on the initial sample. The guarded choice changes four queries for the better and eight for the worse in confirmation. Its change from the frozen ranking is −0.00750 [−0.02188, +0.00429]. This evidence supports measuring order sensitivity separately from final-systems improvement. The test retains the frozen system and stops before official dev. [Setwise prompting](https://doi.org/10.1145/3626772.3657813) · [Matched prompt logits, orderings and scores](../results/analysis/setwise-matched-20261126.tar.gz)

## Discount two votes produced by the same checkpoint

The two evidence votes use the same model and text. Their reciprocal-rank contributions have median within-query Pearson correlation 0.998 on the initial 186. Keep the prior and pointwise votes and the post-fusion first-position rule fixed. Combine the raw and verified evidence contributions by averaging, taking the larger per document, or dividing their sum by one plus their nonnegative within-query correlation. These variants score 0.33420, 0.33255 and 0.33462 against 0.33821. Every fold chooses the existing aggregation. This intervention finds better performance with the original weight on the evidence source in this cascade. [Vote-dependence replay and ranking logs](../results/analysis/dependent-votes-20261126.tar.gz)

## Protocol for new evidence models

Compare a model with a different training objective in the same 14-document shortlist. Use the same fixed question, two evidence excerpts, and original consensus. With Rank1-1.5B, measure the true-minus-false logit from the pinned checkpoint with an empty reasoning prefix and with up to 192 generated reasoning tokens. With DiVeR-Retriever-4B, use its published embedding instruction and last-token cosine protocol. Compare full-question scoring with the average of normalized full-question and strongest clause scores. Each proposed arm adds one equal fifth vote before the existing first-position safeguard. These comparisons test controlled score contributions within the RETECO cascade. [Rank1 model and protocol](https://huggingface.co/jhu-clsp/rank1-1.5b/tree/0b513ca87e0de16ffe17196c6746f28cafaad26b) · [DiVeR model and protocol](https://huggingface.co/AQ-MedAI/Diver-Retriever-4B/tree/ff7f8bc8d1827734dcf2bbfab99f065be618069a)

Eligibility for the reserve requires positive fold-wise gain on the 186, a positive minimum leave-one-domain-out effect there, and positive gain with the same rule on the 155 after omitting any single domain. If both model-specific choices qualify, rank them by the initial fold-wise gain. Score only the selected rule and frozen comparator on the reserved 191. Keep that same rule for the next official-dev comparison if it improves on the 191 and stays positive after omitting any one represented domain. The eight reserve domains provide an additional check on generalization, alongside the 13-domain initial population. Published public-dev evaluations record a paired interval and the smallest leave-one-domain-out gain for the fixed comparison.

The 191-question output uses the same out-of-fold first-stage source used by the 155 confirmation rows. Its preliminary 60-document pool and model scores are generated before relevance evaluation. The original 186 records train the fixed repair, and the 191 are scored only at confirmation.

### Rank1 initial budget and completion calibration

On the initial 186 questions, five-vote nDCG@10 is 0.33198 with the empty reasoning prefix and 0.33685 with a generated prefix capped at 192 tokens. The frozen reference scores 0.33821. The matched effect of the generated prefix is +0.00487 [−0.00068, +0.01130]. On the embedded 65 questions with the exact earlier starting system, the generated-prefix score is 0.33843 against the reference at 0.33558. Fold-wise selection across all 186 scores 0.33675. The 192-token variant stays at the initial stage.

Only 206 of 2,604 generated continuations emitted the model's closing thought token by the cap. For a completion calibration, choose one query per training domain by a fixed hash and reuse all 14 candidate passages. On these 182 inputs, completion rises from 13 at 192 tokens to 180 at 768 tokens. Use completion counts alone for this budget decision and rerun the same 186-query comparison with a maximum of 768 tokens. Keep the earlier 192-token scores, the candidate set, and the model revision fixed for the comparison. [Initial rankings, logits and completion calibration](../results/analysis/rank1-budget192-20261126.tar.gz)

### DiVeR on the fixed evidence shortlist

As an extra vote on the same 14 passages, DiVeR-Retriever-4B scores 0.33712 from the full question and 0.33740 when incorporating the best-matching question fragment. The 186-query frozen reference is 0.33821. The matched fragment change is +0.00028 [−0.00218, +0.00284]. Foregoing the additional vote remains the pooled choice. Selection across folds gives 0.33599 and a minimum leave-one-domain-out change of −0.00259. On the 65 queries with the exact earlier starting system, the fragment score is 0.33895 versus 0.33558. The full sample and fold-wise comparison govern promotion. These are fixed-shortlist results for a component of the released DiVeR system. [Matched rankings and cached DiVeR similarities](../results/analysis/diver-matched14-20261126.tar.gz)

A separate candidate-selection comparison asks whether DiVeR is useful before the 14 passages have been chosen. Keep the prior ten and the strongest existing candidate from each of the two retrieval tiers. In each tier, compare the remaining original candidate against the eligible alternatives under the encoder. The two treatments differ only in the input query representations. One uses the full question, and one also considers its literal fragments. Both keep 14 total grades and use the original graders and final rank fusion. Freeze rules on the 186 and use the same transfer gates before scoring any new official-dev comparison.

#### Keep the reading budget fixed and reward evidence the protected set misses

For the same two open slots, compare full-query similarity, sensitivity to any question fragment, and the additional support a candidate gives beyond the twelve protected documents. Normalize the encoder similarities for each fragment over the 60 candidates. The protected set's best similarity supplies the reference for that fragment. For each candidate, compare its largest positive increase across fragments, combined equally with its normalized full-question similarity within the tier. Refresh the protected scores after filling the first open slot. Use the same 14 evidence grades and the same final reranking policy in each arm.

| Ideal order within the 14 grading slots | Initial 186 | Disjoint 155 |
| --- | ---: | ---: |
| Current evidence shortlist | 0.55238 | 0.52618 |
| Best possible choices in the two open slots | 0.66115 | 0.64741 |
| Queries with an available improvement | 57 | 56 |

The two-slot difference is +0.10877 [0.08246, 0.13751] on the first sample and +0.12123 [0.08874, 0.15665] on confirmation. This oracle uses the training judgments to choose the two documents and ideal ordering. The actual selection compares the encoder criteria using only input text. [Two-slot bounds and per-query audits](../results/analysis/admission-two-slot-oracle-20261126.tar.gz)

After the 768-token rerun, 2,574 of 2,604 outputs reach the closing marker. The 2,604 paired direct-control logits are bitwise identical to the earlier run. The reasoned vote reaches 0.33588, and the fold-wise choice reaches 0.33492. Increasing the budget changes the reasoned vote by −0.00098 [−0.00565, +0.00299]. The full reference remains 0.33821. On the exact baseline subset of 65, the longer-budget score is 0.33781 against 0.33558. The completed reasoning responses leave this fifth vote below the reference on the planned 186-query population. [Longer-budget paired scores and rankings](../results/analysis/rank1-budget768-20261126.tar.gz)

The wide-pool study applies the long-standing idea of explicitly diversifying by subquery to the allocation of expensive evidence grades. It tests the placement of this decision before final ranking, with the same number of graded documents and direct relevance controls. The question fragments come from the observed query text. [xQuAD](https://terrierteam.dcs.gla.ac.uk/publications/ecir2010_rodrygo_div.pdf) · [SetR](https://aclanthology.org/2025.acl-long.861/)

The fixed 191-query baseline is established before using its relevance judgments. The SHA-256 of its final-rank predictions is `c9d50fe91974918543f59a97c5f998d0dd1ddf30ca901e432894124d92945c10`. The consensus and 14-document manifest digests are `6df364b963f33be6cf3772523e60256dbf43798a967452188d05bfd1a86611a4` and `bf2cc009790303a4900e0a7c6ebae28ab6a6184b7d2cead75947b599c627a406`. Replaying the 155-query repair, consensus and final ranks with the same reconstruction code gives byte-identical outputs to their previous versions.

#### Measured two-slot selections

The full-question, clause, and marginal-support selectors achieve 14-document ideal scores of 0.55055, 0.53948 and 0.54653 on the initial 186. The existing shortlist has ideal score 0.55238. Reuse 213 candidate grades and obtain 264 new grades for the union of selected documents. Each configuration still uses 14 grades per query.

| Selection with the same final ranking procedure | Initial 186 nDCG@10 |
| --- | ---: |
| Frozen ranking | 0.33821 |
| Full-question selection | 0.33854 |
| Best-fragment selection | 0.33664 |
| Selection by marginal fragment support | 0.33935 |
| Selection rule chosen on other folds | 0.33664 |

Marginal selection adds 0.00114 [−0.00394, +0.00685] as a fixed policy. Across folds, the chosen policies decline by 0.00157 [−0.00450, +0.00045]. Their smallest leave-one-domain-out change is −0.00178. The 155 have cached text-only embeddings, with their final comparisons held back by the initial criterion. [Wide candidate logits, shortlisted IDs, judgments and paired rankings](../results/analysis/diver-wide-held-20261126.tar.gz)

## Teach the same scorer which part of a question a document supports

Use the official training steps to supervise opposing document preferences for two literal subquestions. A step maps to an interrogative from the original question when the best normalized term overlap is at least 0.17, at least 0.055 above the second-best, and includes two shared non-stopword tokens. Keep pairs of steps mapped to different requests with distinct labeled evidence. Both selected documents must also be labeled relevant for the complete question. Pick one document from each step's exclusive evidence with a deterministic hash. This yields 72 comparisons over six domains, including 60 in History. A document receives credit for the evidence annotated for the requested step under this training convention.

Exclude all 532 query IDs from the three earlier evaluation samples before assembling either training arm. Use 371 common full-query groups, sampled from previously encoded 30-document lists. For each of the 72 matched comparisons, create two extra groups with identical documents across arms. The whole-question control gives each of the two documents full-query relevance credit in both groups. The contrast arm highlights one literal question fragment at a time and asks the model to score the fragment's evidence using the complete question for time and entity context. The positive label switches with the highlighted fragment. Each arm has 515 groups.

Use the identical pinned Yuanbao-Reranker-0.6B checkpoint, optimizer, LoRA size, one epoch, seed, rank loss, 1,024-token limit and corpus excerpts. Evaluate frozen weights, the whole-question control and the contrast-trained weights on the same 14-document pools. Each model scores the full question and each literal fragment with the same excerpt selection. Compare the full-score vote with a vote averaging the normalized full score and strongest normalized fragment score. Only the contrast-trained fragment vote is eligible for transfer. Use the same fold and domain gates before the disjoint 155, the reserved 191 and any later official-dev test.

Both adapters train 589,824 parameters in 129 updates. On the 19 held-out step pairs with the same document inputs, the frozen model, control adapter and step adapter each produce one correct preference reversal and 20 correct single-clause preferences out of 38. The step training therefore leaves the observed clause-discrimination behavior unchanged. The base, control and step adapters' fragment votes score 0.33322, 0.33330 and 0.33322 on the 186-query ranking task. The reference scores 0.33821 and remains the choice in all three folds. Both training inputs are reproduced byte for byte from the documented preparation. [Paired scores, fixed probe IDs and training logs](../results/analysis/step-contrast-held-20261126.tar.gz)

## Search the corpus for evidence missing from the current top results

Ask the 4B selector to choose one numbered, literal request from the complete question. Compare its choice when it can see the four leading retrieved excerpts with its choice from the question alone. Both comparisons use the same four-way instruction and selector weights, and each initiating view searches the same corpus index. The initial full-query mxbai ranking serves as a third control. A selected request keeps up to 32 words from the query's first line and the original wording of the selected fragment. Single-request inputs use the full question for each view.

Retain the old 60 candidates. Each retrieval input supplies four new documents from the unchanged corpus index. Compare access to judged positives and the ideal score with one eligible admission at positions three through ten. Only the training-side samples supply judgments for the candidate diagnostic. The outcome isolates what evidence feedback contributes to finding a genuinely new candidate before spending the expensive grading budget. Interleaved retrieval and generation motivate this study of a bounded retrieval action. [IRCoT](https://aclanthology.org/2023.acl-long.557/) · [Active Retrieval Augmented Generation](https://aclanthology.org/2023.emnlp-main.495/)
