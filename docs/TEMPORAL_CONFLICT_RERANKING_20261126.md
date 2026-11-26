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
