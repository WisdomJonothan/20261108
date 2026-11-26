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

The original-query control uses the same question loader as the archived dense run. The preliminary replay used the reranker query text and recovered 66.13% of the four documents proposed by the archived dense order. The corrected control recovers **99.60%**, with an exact four-document ordering on 178 of 186 questions. The preliminary artifacts remain separate from the final comparison.

Seeing the four excerpts changes the selected fragment on 31 of the 148 questions with multiple eligible requests. Under the same four-candidate budget, each method searches the full domain corpus. The 8B scorer chooses one candidate per source for detailed evidence grading. Admit at most one new document at ranks three through ten with a matching quotation, a minimum expected grade of 2, a 0.5 advantage and a higher pointwise score than the displaced document.

| Retrieval view | Queries with a new labeled positive among four | Queries with one in the leading proposal | Ideal one-admission score | Actual nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Complete question | 5 | 3 | 0.34360 | 0.33815 |
| Request selected from the question | 10 | 5 | 0.35078 | 0.33808 |
| Request selected after reading existing excerpts | 7 | 2 | 0.34680 | 0.33808 |

The frozen ranking scores 0.33821. The three actual policies each improve one question and lower one. Fold-wise selection scores 0.33745. The separate candidate and grade stages expose where the additional labeled evidence falls away. [Aligned retrieval candidates, initial and final proposals, matched scores and rankings](../results/analysis/evidence_feedback_20261126.tar.gz)

## Use a separate reasoning encoder on two broader candidate sources

Form the union of the first 200 documents from each of the archived mxbai and BGE lists and the original 60. DiVeR produces normalized document and query embeddings using its instruction protocol and last-token pooling. Each document representation is shared across every question in its domain and shard. Each of the three sources supplies its best four candidates outside the original pool. Compare the two constituent retrievers with DiVeR ranking the union. The shortlist diagnostic uses matched budgets and the same allowed ranks for candidate admission. Follow-up grading reuses the evidence and pointwise checks from the feedback experiment. [DiVeR encoder and training objective](https://huggingface.co/AQ-MedAI/Diver-Retriever-4B/tree/ff7f8bc8d1827734dcf2bbfab99f065be618069a)

| Nested candidate constraints on the initial 186 | nDCG@10 |
| --- | ---: |
| Frozen system | 0.33821 |
| Ideal order of the current ten documents | 0.49799 |
| Ideal order with the 14 evidence documents available | 0.55238 |
| Ideal order with the original 60 documents available | 0.67516 |
| Ideal order in the union of the two 200-document lists and the original 60 | 0.81636 |

Of 57 zero-score queries, 9 already have a labeled positive among the 14 evidence documents, 26 have one in the 60, and 40 in the broader union. These nested diagnostics separate early candidate availability, allocation of evidence grades, and final position.

| Source for four new candidates | Queries with a labeled positive among four | Queries with one in the 8B proposal | Ideal one-admission score | Actual nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| mxbai | 5 | 3 | 0.34360 | 0.33891 |
| BGE | 7 | 5 | 0.34661 | 0.33821 |
| DiVeR on the union | 17 | 7 | 0.35730 | 0.33884 |

Choosing a source on other folds yields 0.33821, equal to the frozen system. Each of the two positive fixed-source outcomes improves one query. DiVeR supplies labeled positives on ten additional queries before the 8B shortlist, motivating a direct test of the grading allocation.

#### Grade four proposals before allowing one admission

Keep the same four proposals per source, 8B scores, corpus excerpts, and admission rule. Compare grading all four against grading only the 8B favorite. The frozen top ten gives the same incumbents and existing evidence grades. Reuse the 500 already available new grades for both controls. Generate evidence grades for the additional proposals alone. Each source can replace at most one document in ranks three through ten. Apply the same fold and domain gates before any disjoint confirmation. This study tests the cost of screening before evidence grading while holding the candidate sources and final decision fixed.

The larger budget adds 1,440 distinct grades across the three sources and reuses all 500 earlier grades. With DiVeR candidates, the number of queries with a verified quote from an available labeled positive increases from 6 to 12. The half-point grade advantage reduces these counts to 1 and 5. With the final pointwise check, 1 and 3 queries remain eligible. The larger budget improves two rankings and worsens one. Its macro score is **0.33988** versus 0.33821, a change of +0.00167 with a paired 95% interval of [−0.00133, +0.00532]. All three folds choose DiVeR with four grades, and the smallest leave-one-domain-out change is +0.00049.

An independent replay uses the organizer scorer on all 186 rankings and verifies the frozen comparator, fold selection, and macro changes. The positive initial gate triggers confirmation on 155 disjoint questions with DiVeR and four grades fixed. The question-level results and intermediate candidate losses are included in the archive. [Nested candidate headroom, all proposals, grades and rankings](../results/analysis/native-wide-held-20261126.tar.gz)

The fixed policy reaches **0.32227** on the disjoint 155 compared with **0.32381** for their frozen ranking. It improves one question and reduces two. The change is −0.00154 with a paired 95% interval of [−0.00557, +0.00174]. The smallest leave-one-domain-out change is −0.00262. The organizer scorer independently reproduces the rankings and results. This route stops before the reserve and official dev. [Disjoint candidate and score archive](../results/analysis/native-wide-confirm-20261126.tar.gz)

## Compare a stronger evidence judge under the same retrieval inputs

Use the pinned Qwen3.5-9B model in text-only, non-thinking mode. Keep the 14 document identities, dispersed excerpts, ordinal rating rubric, quote verification and 80-token explanation limit. Measure standalone direct and explanation-based judgments for both model checkpoints. These measurements expose changes in score quality at the same evidence scope.

For the deployed ordering comparison, either replace both original evidence votes with the new judge's explanation and verified votes or add only its verified vote. Keep the current non-evidence votes and original first-position guard fixed. A replay substituting identical stored grades reproduces all 186 frozen rankings exactly. Use the earlier fold and disjoint-population gates for any new submission policy. [Pinned Qwen3.5 model](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)

| Fixed 186-query comparison | 4B | 9B |
| --- | ---: | ---: |
| Standalone direct ratings | 0.30342 | 0.30485 |
| Standalone ratings after the explanation | 0.32601 | 0.29222 |
| Exact quotation checks passed out of 2,604 | 1,516 | 1,024 |

The four-shard 9B inference completes the full 2,604 matched pairs. It places a quote and an explanation on the requested lines in every response. Exact quotation verification still passes less frequently. Replacing both evidence votes scores 0.33365, and adding the verified vote scores 0.33706. The frozen ordering scores 0.33821. Fold-wise choice scores 0.33683 and its smallest leave-one-domain-out change is −0.00181. The same-grade replay also exactly reproduces the frozen rankings on the 155, the 191 and official dev without accessing their relevance judgments. The stronger-judge comparison stops after the initial 186. [Same-document score and ranking archive](../results/analysis/teacher-held-20261126.tar.gz)

## Test whether a copied citation supplies the relevance judgment

Use the frozen 14-document pools and the same 4B rating rubric. For a verified quotation that appears once, compare three matched document views. Present the quote alone, remove the quote from the original excerpts, or remove a disjoint span of the same word length chosen by a deterministic hash. The paired removal controls for generic effects of deleting text. Quotes without a disjoint control keep their original grades. This adapts rationale sufficiency and comprehensiveness to a temporal retrieval decision. [ERASER](https://aclanthology.org/2020.acl-main.408/) · [Explanation metric controls](https://aclanthology.org/2024.findings-eacl.88/)

Apply the intervention to the verified evidence vote. The sufficiency arm caps a generated grade at the quote-only rating. The paired arm also caps it at the original direct rating when removing the quotation lowers support no more than removing the comparison span. Keep the other three votes and the final first-position rule fixed. Measure paired context changes and final macro nDCG on the initial 186. Use the existing transfer gates for any retained policy.

The matched removal is available for 1,304 document-query pairs across 181 queries. Removing the quotation reduces the score more than removing the comparison span in 939 pairs (72.0%). The mean relative drop is 0.451 on the original zero-to-three scale, with 0.473 among 171 labeled positives and 0.448 among the other 1,133 pairs. The quote-only cap and paired cap score 0.33633 and 0.33630, compared with 0.33821 for the frozen ranking. Both change 19 queries for the better and 26 for the worse. Selection on the other folds scores 0.33642 with a minimum leave-one-domain-out change of −0.00255. Organizer scoring reproduces the query results. The quote intervention exposes text the model uses, and the ranking replay measures the separate value of using that signal to change the final ordering. [Masked views, paired scores and rankings](../results/analysis/masked-quote-held-20261126.tar.gz)

## Compare pooled and temporal token matching with the same encoder

Use the released BGE-M3 checkpoint to score the same candidate union and document prefix as the wider DiVeR experiment. Compare its dense cosine score, its standard mean MaxSim token score, and a MaxSim score with double weight on query tokens overlapping a literal year or an explicit temporal ordering cue. The two token-level arms share exactly the same contextual token similarities. The last arm tests where the model spends its matching weight. Compare equal four-document proposals outside the frozen 60 and use the same one-admission oracle and paired final-ranking gate. DiVeR, mxbai and BGE proposals in the earlier experiment provide archived controls for the same union. [M3-Embedding](https://aclanthology.org/2024.findings-acl.137/) · [Pinned BGE-M3 checkpoint](https://huggingface.co/BAAI/bge-m3/tree/f11a3f18ef0f9740f5268d633d025d11b33e1e82)

The implementation reproduces the publisher's two dense and MaxSim example scores within 0.001. The score emphasizes temporal tokens on 123 of the initial 186 queries, and it changes the order or membership of the four offers for 30 queries. With the same final one-admission checks, the standard and temporal token variants produce identical final scores and improve the same two queries.

| Proposal score from the same BGE-M3 weights | Queries with a labeled positive | Ideal one admission | Actual nDCG@10 |
| --- | ---: | ---: | ---: |
| Dense cosine | 10 | 0.34914 | 0.33894 |
| Mean MaxSim | 9 | 0.34835 | 0.33957 |
| Temporally weighted MaxSim | 9 | 0.34835 | 0.33957 |

The frozen reference and fold-wise choice both score 0.33821. Each token-matching arm has a fixed gain of +0.00136 and skips disjoint confirmation after the cross-fold result. The organizer scorer reproduces the candidate rankings. [Matched token proposals, 1,215 graded pairs and ranking outcomes](../results/analysis/m3-held-20261126.tar.gz)

## Compare evidence allocation before fusion with admission after fusion

The one-admission experiment exposes a loss between finding a labeled positive and clearing the fixed margin against an incumbent. Compare an earlier intervention that gives a new candidate an evidence vote before the rankings are fused. Preserve the ten incumbent documents and the first proposal from each original retrieval tier. For the two remaining positions in the 14-document shortlist, let the original candidates compete with the one or two largest 8B scores among the same four DiVeR offers. Fill a position only when an outside candidate wins by the matched 8B score. Keep the original position of any retained candidate.

The two policies change which 14 documents receive the original evidence grades. They preserve the old non-evidence votes, fusion and first-position check. Existing model outputs enable a reproducible replay without additional inference for the first 186. The 155 have already provided a result for the post-fusion intervention, so this experiment preserves the independent 191 for any eligible earlier-allocation rule and keeps official dev sealed. Apply the same positive cross-fold and leave-one-domain-out criteria before those transfers.

Both placement comparisons reproduce the frozen ranking when the original shortlist is retained. Opening one slot scores 0.33651 versus 0.33821 and changes six questions for the better and five for the worse. Opening two slots scores 0.33665 and changes eight for the better and six for the worse. The larger negative changes account for the lower macro score. The three folds each retain the original policy. This replay uses cached judgments, and it stops before any new confirmation scoring. [Fixed-budget candidate allocations and locked rankings](../results/analysis/earlier-slots-held-20261126.tar.gz)

## Screen a deeper union with a small cross-encoder

Use a pinned 0.6B Yuanbao relevance model with the same dispersed evidence windows for every pair. Score the complete union of the two depth-200 lists and the original 60. The shallow comparison uses only the same two lists through depth 50 and the original 60. Each arm proposes the four highest-scoring documents outside the original 60. Shared candidates use identical scores. Use the established 4B evidence extraction and 8B pointwise checks to evaluate one admission, with the same four expensive grades per arm. Keep the training-side selection and confirmation gates. This determines whether spending more cheap cross-encoder comparisons earlier produces improvements that survive the late admission rule. [Pinned Yuanbao reranker](https://huggingface.co/Yuanbao-Search/Yuanbao-Reranker-0.6B/tree/0101299a7937a016848d8d55b092f6e31cfa312d)

The larger scan averages 290.9 distinct candidates and the nested shallow comparison averages 91.4. Permit the scorer's existing 4,096-token limit to clip exceptional overlength prompt bodies while preserving the full verdict suffix. Record the count of these inputs and keep the shared candidates' scores identical across depths. As a diagnostic, calculate the small cross-encoder's standalone order within the original 60, shallow union, and deep union. Eligibility for transfer remains based on the final ranking with the same admission policy.

The complete scan scores 54,103 pairs and clips one overlength input. Across the two arms, the 4B and 8B models each score the same 1,172 distinct proposed pairs.

| Depth per parent list | Queries with a labeled positive among four | Ideal one admission | Actual nDCG@10 |
| --- | ---: | ---: | ---: |
| 50 | 13 | 0.35525 | 0.33650 |
| 200 | 20 | 0.36334 | 0.33994 |

The frozen score is 0.33821. All three folds choose the deeper scan, which improves two queries and harms zero. The macro change is +0.00173 with a paired interval of [0.00000, +0.00440] and a minimum leave-one-domain-out change of +0.00087. This fixed policy advances to the 155-query confirmation. On the 165 queries with step labels, the baseline, shallow, and deep step coverage are 61.72%, 61.45%, and 61.91%. The small cross-encoder's standalone scores in the original 60, shallow, and deep pools are 0.27713, 0.27224, and 0.28140. Deep versus shallow changes this standalone diagnostic by +0.00916 [−0.00050, +0.02086]. [Nested depths, raw score cache, and ranked controls](../results/analysis/deep-held-20261126.tar.gz)

The locked deeper policy scores **0.32455** on the disjoint 155 versus 0.32381 for their frozen baseline. The difference is +0.00074 [−0.00696, +0.00965], with two higher and four lower query scores. One of the four Law queries supplies the largest domain gain (+0.04880). Without Law the change is −0.00363, the smallest leave-one-domain-out contrast. The fixed policy stops at this gate. On the 135 questions with step labels, coverage increases from 58.66% to 60.07%. These questions provide a second comparison of evidence availability and final precision under the same policy. [Disjoint scores, rank outputs and frozen-policy replay](../results/analysis/deep-confirm-20261126.tar.gz)

### Replace the absolute margin with cross-encoder agreement

To test the grade-margin bottleneck observed with DiVeR, keep the four depth-200 proposals and exact quotation requirement. Require the candidate to have a grade of at least two and a positive evidence-grade advantage over the incumbent. Both the 0.6B scan and the 8B reranker must also prefer the candidate. This permits a positive evidence difference below half a point. Evaluate one fixed agreement policy against the frozen ranking and record the earlier half-point rule on these same candidates as a control. Choose the eligible change only on training-side folds and keep the independent reserve and final official-dev gate.

The agreement rule scores 0.33760, with one improvement and two declines. Fold-wise selection scores 0.33691. The half-point control exactly replays 0.33994 with the same candidates. Confirmation uses the original half-point decision. The organizer scorer independently reproduces both sets of rankings and macro changes.

### Spend the existing 60-candidate 8B budget on a broader shortlist

Use the 0.6B depths study's existing scores to select a 60-document pool from the broader union. Protect the frozen top ten, then add the 50 highest-scoring remaining documents. Use identical 8B inputs and scores for candidates shared with the original 60. Compare the standalone 8B ranking on each 60-document pool. As a final-system comparison, replace the pointwise vote with the top ten from each 8B ranking. Keep the other three votes and the existing first-position rule. The broader shortlist is the sole eligible new policy. Apply the same transfer gates before scoring on official dev. This compares replacing an entire vote with the earlier approach of accepting one verified admission at the end.

The ideal score of the revised 60-document pool rises from 0.67516 to 0.75363. The pool gains judged positives on 49 queries and loses some on 11. The 8B model grades 4,429 newly exposed pairs. The standalone original and wider pools score 0.28252 and 0.27132. Replacing the current pointwise vote with either standalone list yields 0.32904 and 0.32627, compared with 0.33821 for the frozen ranking. All three folds retain the original policy. The fixed-budget study shows a larger candidate ceiling alongside a lower end-to-end score for standalone replacement.

### Preserve the original positional vote and test its normalization scope

The 30-candidate pointwise vote combines ranked-position discount with the 8B scores at equal standardized weight. Replaying this rule reconstructs the original vote in all 186 cases. Retain the original first 20 candidates and replace positions 21–30 with the first ten new candidates from the matched wider pool. Use the same 30-document rule and existing 8B scores. Compare normalizing within the updated 30 with using the original thirty's mean and standard deviation. Hold the candidate IDs, other three votes, and first-position rule fixed. This tests whether changing the candidate distribution disrupts the original score calibration. The original three-fold and disjoint confirmation criteria apply.

The 8B score standard deviation falls on 174 of 186 queries, with a median new-to-original ratio of 0.777. The denominator changes the weight of the 8B signal relative to the position prior. The centering term cancels when comparing documents from the same query. The two scales produce different final rankings on 88 queries.

| Pointwise vote within the fixed consensus | nDCG@10 |
| --- | ---: |
| Frozen reference | 0.33821 |
| Updated candidate pool, local score scale | 0.33878 |
| Updated candidate pool, original score scale | 0.33943 |

The original scale gains +0.00122 [−0.00153, +0.00431] over the frozen reference and +0.00064 [−0.00049, +0.00200] over recomputing the scale locally. It improves 11 queries and lowers eight. All three folds choose the original scale. Its smallest leave-one-domain-out improvement is +0.00035. Lock that rule and evaluate it on the 155 independent of the three selection folds. The organizer scorer reproduces both the standalone-vote and position-prior comparisons. [Matched wide scopes, 8B scores, score-scale diagnostics and rankings](../results/analysis/wide-vote-held-20261126.tar.gz)

Bruch et al. compare normalization functions and learned convex-fusion weights. Chauhan and Agarwal measure query-level losses from lexical and dense fusion on seven BEIR datasets and follow the losses through an equal-budget reranking stage. The RETECO comparison fixes the expanded document list and its score values, then intervenes on the candidate set that supplies the score standard deviation to a later fusion stage. That isolates a query-dependent change in effective weight caused by candidate selection even when the nominal combination is fixed. [Analysis of Fusion Functions](https://arxiv.org/abs/2210.11934) · [The Fusion Tax](https://arxiv.org/abs/2610.30612)

#### Disjoint confirmation

The locked original-scale rule scores **0.32688** against 0.32381 on 155 disjoint questions. It adds +0.00307 [0.00039, 0.00631] with thirteen higher and five lower query scores. Its minimum leave-one-domain-out difference is +0.00209. The score standard deviation shrinks on 136 of 155 questions, with a median ratio of 0.804, and the two scales change 61 final lists.

The local-scale control scores 0.32856 on the 155. Its lead over the fixed scale is +0.00168 [0.00012, 0.00401], reversing the ordering observed in the initial 186. Both rules improve on the frozen comparator in this confirmation. Retain the preselected original scale and evaluate only that rule and the comparator on the reserved 191. Organizer scoring independently reproduces the confirmation results. [Disjoint rankings and scale diagnostics](../results/analysis/wide-vote-confirm-20261126.tar.gz)

#### Reserved 191

The original-scale rule reaches 0.28281 versus 0.28212 on the reserved eight-domain sample. The difference is +0.00069 [−0.00188, +0.00352], with eight improved and seven declined queries. Removing History yields −0.00024, the smallest leave-one-domain-out change. The locked rule therefore stops before official dev. The reserved archive contains only the two rankings used for relevance scoring. [Reserved IDs](../results/analysis/passage-and-conditional-replay-20261126.tar.gz) · [Reserved predictions and replay](../results/analysis/wide-vote-reserve-20261126.tar.gz)

### Keep final positions that agree under both score scales

Use the same expanded 30-document pool and fixed ranking code from the scale comparison. Keep a document in a proposed final position whenever both scale variants put that document at the same position. Fill the other positions in the original final order, skipping documents that already have an assigned position. This gives a fixed agreement policy with one eligible treatment and the frozen comparison. The first gate uses the original 186. If it advances, apply the fixed rule on the 155. The earlier reserve has supplied a result for the original-scale rule. Any further independent gate uses query IDs outside all three previous populations.

The fixed agreement rule scores 0.33898 with ten higher and seven lower query scores. Selection across folds scores 0.33797, a change of −0.00024, with a smallest leave-one-domain-out change of −0.00038. It stops before the independent confirmation. [Matched agreement rankings](../results/analysis/scale-agreement-held-20261126.tar.gz)

## Train the same 8B reranker on positives from different candidate distributions

The nested depth scans find additional judged positives and the final scoring stage realizes a small share of that opportunity. Use the same pinned 8B reranker and identical ranking loss in a matched positive-sampling comparison. The two training arms share their queries, four label-zero candidates per group, group order, optimizer, and length limit. Common groups use one positive from the initial mxbai 60 in both arms. A matched additional group uses another observation of the near positive in the control and a qrel-positive document from outside the mxbai 60 in the treatment. The wider positive is chosen from the union of the depth-200 lists. Two benchmark-zero candidates come from the original list and two from the broader part. Exclude documents annotated as evidence for any training step from these weak negatives.

First exclude all 532 IDs in the previous 186, 155 and 191 studies. Reserve up to 60 remaining queries per domain from the archived 969-query candidate manifest by a fixed domain-and-query hash. Exclude them from training as well. Cap the common and matched groups at 140 and 70 per domain using fixed hashes to moderate the History concentration. Train both arms for one epoch with rank-8 LoRA in the last eight layers, a peak learning rate of 0.00002, seed 2954 and the existing listwise loss. A memory-only smoke test on the longest prepared examples selects the shared 1,024-token limit or a 768-token fallback before any ranking readout.

Evaluate the frozen 8B scorer and both adapters on the same 30-document pool containing twenty existing candidates and ten from the wide pool. Use local standardization within this identical pool for each checkpoint, the same vote fusion and the same protected first-position rule. Score the adapted model against both the frozen competition ranking and the matched control. Select the wider-positive adaptation only if it improves over both on the initial training-side folds and retains positive leave-one-domain-out effects. Use the 155 and the new disjoint reserve for any transfer. The original 191 remains an earlier reported comparison.

The preparation yields 342 groups per arm, with 241 identical common groups and 101 comparisons with a changed positive. All 342 negative token lists match exactly. The new reserve has 109 IDs across Genealogy, History, History of Science, and Politics. The longest-input smoke test exceeds memory at 1,024 tokens, and the prescribed 768-token fallback succeeds. Both adapters complete 86 updates with 1,900,544 trainable parameters and 21,655 MiB peak PyTorch allocation. The control restarts with identical settings after a concurrent allocation occupies its first device. The complete matched result uses the successful full-epoch control.

| Scorer on the same enlarged candidate list | Standalone nDCG@10 | With the existing consensus |
| --- | ---: | ---: |
| Frozen 8B | 0.28471 | 0.33878 |
| Near-positive control | 0.28523 | 0.33985 |
| Wide-positive adapter | 0.28284 | 0.33886 |

The frozen competition ranking scores 0.33821. The matched target-control change in final nDCG is −0.00099 [−0.00253, −0.00006]. Fold-wise choice scores 0.33809, with a smallest leave-one-domain-out change of −0.00037. The portion of supervision that moves positive examples into the wider retrieval distribution leaves the matched system below the near-positive control. This run stops at the initial 186. [Matched group audit, training logs and scores](../results/analysis/wide-adapter-held-20261126.tar.gz)

## Compare the first document and one challenger in both presentation orders

Keep the frozen ten documents. For each query, choose the strongest existing 8B vote among ranks two through five with an exact 4B quotation and evidence grade of at least two. The competing document and current first result receive the same visible excerpts and full question in both orders. Use the same pinned 9B judge that scored the ordinal rubric. Compare an immediate choice of A or B with a choice after up to 128 generated explanation tokens. An edit requires both presentations to prefer the challenger under the same reading protocol. Swap only the two positions. Evaluate both balanced policies on the 186 and use the existing transfer gates. The decision scope tests a high-discount correction while jointly controlling position bias.

The comparison nominates a challenger on 163 of 186 questions. Choosing when to swap from the official training judgments gives an ideal score of 0.37100, with 29 opportunities to improve and 29 opportunities to harm. Reading the same pair in both orders gives consistent winners for 40 direct comparisons and 47 explanation-based comparisons. Requiring agreement accepts 13 or 10 swaps. The direct and explanation policies score 0.33863 and 0.34052, versus 0.33821 for the frozen ranking. Explanation-based swapping improves two questions and lowers none. Fold-wise selection remains at 0.33821. The original agreement policy stays below the transfer gate.

### Remove the common presentation offset

Keep the same paired prompts and model outputs. Define the preference as half the B-minus-A margin in the original presentation minus half that margin in the reversed presentation. A positive value favors the challenger once a common label or position offset cancels. Apply the same rule to direct and explanation-based margins, with zero as the fixed decision boundary. Use the same two positions and the same transfer protocol. This tests whether two-order averaging makes better use of the scored comparisons than requiring both absolute choices to agree.

The direct difference gives 0.34413 on the 186 as a fixed rule, with 11 helped and eight harmed queries. Its paired gain is +0.00592 [−0.00347, +0.01564]. The explanation difference gives 0.33758. Cross-fold selection scores 0.33596, and its smallest leave-one-domain-out effect is −0.00420. Both pair studies end before the 155. [Candidate oracle, both orientations and all rankings](../results/analysis/head-pair-held-20261126.tar.gz)

## Screen the near-positive control checkpoint as a separate fixed candidate

The wider-positive supervision comparison provides a supervised control checkpoint. Screen that existing one-epoch near-positive adapter separately for potential ranking improvement. This follow-up was chosen after inspecting the training-side comparison. Keep the exact same 30 candidates, scores, fusion procedure and checkpoint. Compare only the frozen ranking and the near-positive adapter for policy selection on the 186. If eligible, freeze the checkpoint and policy for the 155 and newly sampled 109 IDs. Use only the eventual locked rule in any official-dev evaluation.

All three selection folds choose the near-positive policy. It scores 0.33985 against 0.33821 across the 186, an increase of +0.00164 [−0.00156, +0.00526]. Its smallest leave-one-domain-out gain is +0.00085. Freeze the trained checkpoint and candidate scope. On the 155, it scores **0.32850** against **0.32381**, an increase of +0.00469 [0.00157, 0.00833] with 19 improved and seven declined queries. The minimum leave-one-domain-out gain is +0.00376. The frozen 8B with the same updated candidate list scores 0.32856. The adapter-minus-frozen difference within that list is −0.00006 [−0.00044, +0.00037]. The added candidate scope and fusion are the measured contributors to the improvement in this confirmation.

Keep the already locked policy and proceed to the 109 training questions held out from all recent policy evaluations and the matched adapter training. Reconstruct the frozen comparator on the 109 with the established pretrained models and the repair fit from the original 186. Add the broader candidate scores before opening the 109 relevance judgments. Organizer scoring independently reproduces the first two populations. [Ranking and checkpoint transfer records](../results/analysis/near-adapter-transfer-20261126.tar.gz)

### Account for the origin and rank of the scored documents

For each saved top-ten comparison, partition the nDCG change into terms for newly admitted positives, positives that leave, and rank movement of retained positives. Use each query's original 60-document scope to tag admissions. The terms sum exactly to the organizer's binary nDCG change and are averaged equally across domains.

| Near-positive policy versus frozen ranking | Initial 186 | Confirmation 155 |
| --- | ---: | ---: |
| Entrants from beyond the original 60 | 0.00000 | 0.00000 |
| Entrants from original positions 31 to 60 | +0.00222 | +0.00327 |
| Entrants from the original first 30 | +0.00070 | 0.00000 |
| Departures of judged positives | −0.00058 | −0.00046 |
| Rank movement of retained positives | −0.00070 | +0.00188 |
| Total | +0.00164 | +0.00469 |

The broader scan recovers original tail candidates that were invisible to the original 30-document pointwise vote. With the frozen scorer on the same revised thirty, these tail admissions contribute +0.00330 on the 155, with +0.00191 from retained rank movement and −0.00046 from departures. The accounting distinguishes candidate exposure inside an existing pool from finding a relevant document outside it. It describes the fixed earlier comparisons. [Decomposition code](../scripts/audit_wide_vote_contributions.py)

## Allocate ten pointwise slots by small-scorer preference inside the original sixty

Keep the frozen 8B, corpus and original candidate identities. In the pointwise vote, reserve the first twenty slots as before. Among original positions 21 to 60, select the ten highest-scoring candidates according to the pinned 0.6B model. Present the selected ten in their original order for the positional prior. Keep the existing scale, the other three votes and first-position rule. This allows a deep candidate to replace one of the existing unprotected ten when the same cheap scorer gives it priority, without drawing from outside the original sixty. The number of candidates beyond the original first thirty is an exposure diagnostic.

Screen only this allocation against the frozen competition rank on the 186. Use the existing three-fold, 155 confirmation, and worst-domain deletion criteria. Any further validation uses a disjoint reserve chosen without reference to the rule's ranking outcomes. The broader-list checkpoint experiment retains its existing 109-query decision. At most that experiment's already locked comparison can open this round of official-dev labels. The extra 60-only allocation asks whether the discovery within the original candidate pool is sufficient under the same 30-document strong-reranker budget.

The new allocation exposes an average of 8.61 candidates from positions 31 to 60 per query. As a fixed rule it scores 0.33827 against 0.33821, with a paired change of +0.00006 [−0.00321, +0.00342] across the 186. It improves 17 queries and harms 13. Fold-wise selection scores 0.33779 and the smallest leave-one-domain-out change is −0.00055. The intervention demonstrates that exposing more of the old tail by the 0.6B preference is easy. The unchanged larger scorer and consensus realize approximately zero aggregate gain under this particular allocation. It stops before confirmation. The organizer scorer reproduces both rankings and the fold-level calculation for all 186 questions. [Fixed rankings and 8B scores](../results/analysis/tail-slots-held-20261126.tar.gz)

### Fresh 109-question transfer

The locked rule scores **0.29771** against **0.29703** for the reconstructed baseline. The macro change is +0.00067 [−0.00540, +0.00789], with 12 higher and 12 lower query scores. By domain, the differences are 0.00000 for Genealogy (3 questions), −0.00022 for History (60), −0.00097 for History of Science (23), and +0.00388 for Politics (23). Leaving out Politics gives −0.00040, the lowest leave-one-domain-out change. The rule stops before official dev. The official reference remains 0.40674 on 519 questions.

The decomposition on the 109 assigns +0.00139 to admissions from positions 31 to 60, −0.00287 to positive departures, and +0.00215 to rank movement among retained positives. Entrants from outside the original 60 again contribute zero. In Politics the +0.00388 is a balance of +0.00978 from retained rank movement and −0.00590 from departures. This comparison identifies protecting relevant incumbents as a separate intervention from finding candidates. Local organizer scoring independently reproduces both locked ranks and the domain deletion check. [109 fixed IDs, score shards and rankings](../results/analysis/near-adapter-reserve-20261126.tar.gz)

## Preserve top-ten membership and change only the order

The locked 109 decomposition attributes the gain in its strongest domain to the movement of retained documents. Test a constrained version of the same position-prior method using the frozen 8B, the first twenty original candidates, and the same ten candidates from the wide pool. The local score normalization and other votes stay fixed. Sort the existing ten output documents by the updated consensus weights and append any unvoted incumbents in their original order. Use the established first-position rule on the result. This fixes membership, so the intervention has exactly zero entrant or departure contribution under the binary nDCG accounting. The unrestricted local-scale reference is descriptive. The protected version is the only candidate in the initial 186-query screen.

Risk-sensitive retrieval already models losses relative to a baseline. This intervention tests a specific source of loss in the late-stage consensus: new scoring evidence can alter the ranking of articles that are already in the ten, and it can separately displace judged evidence. Apply the same cross-fold and domain-deletion gate, then the disjoint 155 if eligible. Treat the original 109 as the source of the hypothesis. Reserve confirmation for unused query IDs. [Wang et al., risk-sensitive retrieval](https://www.cs.cmu.edu/~pbennett/papers/wang-et-al-sigir-2012.pdf)

## Decay contributions near the pointwise vote cutoff

A single pointwise rank change across positions 10 and 11 removes the entire vote contribution in the original hard truncation. Test whether retaining scaled contributions for positions 11 to 29 reduces the loss of relevant incumbents when the pool changes. Keep the three other top-ten votes, the 8B scores and original position prior, local standardization, and the first-position rule. For rank r through 10 the pointwise weight is 1/(60+r). Above rank 10, multiply it by (30−r)/20 until it reaches zero. This changes how much near-boundary candidates contribute without additional model calls. Compare hard and decaying votes on both the original and updated 30. The updated list with decaying contributions is the only new policy eligible to advance. Its fixed and fold-wise effect must pass the 186 screen, with a positive worst-domain deletion effect and an aggregate gain above the matched updated-list hard cutoff. Apply the same confirmation to the 155.

A later confirmation uses the 242 official training topics outside the archived 969 candidate manifest, as selected by topic ID without the 242 judgments. These queries span 13 domains. Their first stage uses the pre-existing all-training-fold ranking with mxbai and BGE candidates, before the existing evidence repair fitted on the 186. The domain shares on this sample range from one IOTA query to 112 History queries. Treat any exposure to these questions during the historical upstream system development as prior exposure, and hold their labels out of the current rule screen. The prior 109 or 191 results do not serve as eligibility evidence for this new rule. No new official score is calculated unless the chosen rule transfers after dropping any one domain on the 242.

## Require evidence for changing top-ten membership

Reuse the unrestricted local-scale vote with the frozen 8B and enlarged candidate list. Keep its updated ordering of shared incumbents. For an incoming result, consider the weakest-evidence incumbent left out of the new list. Admit the incoming document if its verified grade is at least two and beats the incumbent's verified grade by at least half a point. An incumbent with no verified grade has zero measured support. Otherwise restore a displaced incumbent in the proposed slot, retaining the original order among any restored documents. The two grade criteria reuse the established comparison for a change at rank one. Every other vote, the candidate list, and original proof assessments are fixed. Measure the number of admissions retained and score contributions of entries, departures and retained rank movement.

Screen this one membership constraint on the 186, with the unrestricted method as a matched reference and the original competition configuration as the selection baseline. Advance only with a positive fold-wise gain, positive worst-domain deletion gain, and a point estimate above the unrestricted method on the same data. Confirm the fixed candidate on the 155 and then on the 242-domain-complete panel before any new official comparison. It tests whether the evidence already computed for membership changes can preserve the value of a broader pointwise vote.

### Screening results for membership and vote constraints

| Fixed method on the same 186 questions | nDCG@10 | Fold-wise selection |
| --- | ---: | ---: |
| Frozen ranking | 0.33821 | 0.33821 |
| Reorder the original membership | 0.33720 | 0.33821 |
| Taper the original candidate vote | 0.33825 | Descriptive |
| Taper the widened candidate vote | 0.33804 | 0.33770 |
| Require verified evidence for admission | 0.33729 | 0.33821 |

The widened taper has minimum leave-one-domain-out effect −0.00144 after cross-fold selection. The admission safeguard reduces entries from 55 to 26 across the 186. Both it and the pure reorder leave all folds on the original competition policy. These variants stop before confirmation. The score decomposition specifies an observed failure mode. The simple global remedies for it give little benefit under this scorer. The organizer scorer independently verifies every ranking for each comparison. [Membership replay](../results/analysis/protected-membership-held-20261126.tar.gz) · [Taper and 242 reserved IDs](../results/analysis/taper-pointwise-held-20261126.tar.gz) · [Evidence admission replay](../results/analysis/guarded-admission-held-20261126.tar.gz)

## Estimate rank movement and membership gains separately

The position and membership interventions have different outcomes across samples. Compare the frozen ranking with two available actions for each query. One reorders the current ten under the revised vote and the other also permits the proposed membership changes. For a relevant document, the nDCG change exactly separates a weight shift among retained documents from the addition or loss of document membership. Fit one common ridge model for each component across the two actions, using the same fixed six features. The signals encode the discounted count, the two original positional priors, verified evidence grades, and the two model score percentiles. Contributions use only the source documents associated with the component being predicted. Use zero intercept, penalty 32 and equal total fit weight across domains. Choose the action with maximum predicted gain, including zero for keeping the frozen ranking.

Pool the four disjoint populations of 186, 155, 191 and 109 as development data. Evaluate without fitting on the held domain, and separately without fitting on the held population. The coefficient model is eligible for transfer only if both cross-fitted estimates beat the frozen ranking (including every single-domain omission) and both beat the unrestricted local-scale alternative. All transfer decisions use the full eligible frozen fit on the reserved 242. Only the frozen baseline and the chosen action's ranks are opened to those judgments. Report the selected action counts and paired uncertainty. The method tests calibration of two different mechanisms using existing evidence and identical inference budget at transfer. Risk-sensitive ranking is prior work. The matched intervention here is the use of separate targets for these exact contributions to a downstream ranking change. [Wang et al.](https://www.cs.cmu.edu/~pbennett/papers/wang-et-al-sigir-2012.pdf)

### Cross-fitted development result and choice for the 242 questions

| Method on the 641 development topics | nDCG@10 | Change |
| --- | ---: | ---: |
| Frozen reference | 0.31351 | 0.00000 |
| Separated gain model, leave an entire domain out of fit | 0.31475 | +0.00124 |
| Separated gain model, leave an entire population out of fit | 0.31438 | +0.00087 |
| Fixed local-scale vote, frozen 8B | 0.31497 | +0.00146 |

The gain model improves in both cross-fitted evaluations. Their minimum domain-deletion gains are +0.00085 and +0.00051. The two paired intervals are [−0.00025, +0.00287] and [−0.00065, +0.00256]. Both estimates fall below the fixed local-scale vote with identical inference inputs, so the fitted model stops. The simpler matched comparator uses the same position-preserving candidate list and the original 8B checkpoint. Its paired interval on the 641 is [−0.00027, +0.00334], its smallest leave-one-domain-out gain is +0.00114, and its smallest gain after removing a whole population is +0.00071. Alone, the four population changes are +0.00057 (186), +0.00475 (155), −0.00044 (191), and +0.00100 (109).

Freeze only the fixed unadapted local-scale comparator for the 242, chosen from development results. The earlier near-positive adapter and original-scale policy remain stopped at their own reserves. The 242 judgments have not been used in the current rule development. The same exact policy must give positive change on the 242 with positive change under every single-domain omission before opening one official-dev comparison. Use the existing four-vote ranking and first-position safeguard. The acquisition of the 242 reference scores proceeds independently of this choice and requires no reserved judgments. The cross-fitted experiment establishes that the fixed control is stronger than the additional regression with the current evidence features.

Organizer scoring reproduces the three action scores and both sets of cross-fitted predictions on all 641 training topics. The fixed-rule record carries the SHA-256 of that development readout. [Development action ranks, cross-fitted policy choices and source manifest](../results/analysis/separated-gain-crossfit-20261126.tar.gz)

### Quantify which evidence reaches the top ten

On all 641 development topics, the fixed local-scale vote contributes **+0.00209** from judged positive entrants at original ranks 31 to 60 and +0.00006 from original ranks 1 to 30. Documents outside the original 60 supply zero judged positive admissions. Losing relevant incumbents costs −0.00087, and changing the rank of retained positives adds +0.00018. This accounts for the full +0.00146 and localizes most observed benefit to giving an existing lower-ranked candidate a chance in the pointwise stage.

On the 581 development topics with step labels, domain-macro step coverage increases from 57.16% to 57.60%. The paired change is +0.44 percentage points [−0.13, +1.18]. Seven questions improve and three decline. The step outcome uses the supplied evidence annotations, independently of the binary primary nDCG calculation. [Per-population contribution tables and coverage](../results/analysis/wide-vote-development-contributions-20261126.json)

For the frozen rule, score only the previously unscored documents that enter the 30-document pointwise vote. In a dry run, this reduces additional 8B scoring from 3,359 to 660 question-document pairs and gives identical ten-document reference and local-scale rankings on all 109 previously scored queries. The revised 60 still uses the shared inexpensive model, and this change leaves the scoring model and list identities fixed. Apply the same execution optimization to the unopened 242 and any eligible official comparison.

The 242 IDs match the earlier project calibration fold exactly. Its first-stage model was trained on the separate 969 according to the fold manifest. An earlier replacement-gate study used the calibration labels to choose its threshold. The current fixed vote did not use those IDs or judgments for policy choice or fitting. This is an internal transfer check for the current rule, with prior use elsewhere in the project. [Historical fold manifest](../results/replacement_v2/ce_folds.json) · [Prior calibration procedure](../scripts/gate_fit.py)
