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
