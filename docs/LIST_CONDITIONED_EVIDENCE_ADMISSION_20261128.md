# Compare evidence gained and lost by a proposed replacement

The [two-sided retrieval study](CONTRASTIVE_EVIDENCE_RETRIEVAL_20261128.md) offered ten new judged-positive documents on 191 queries. Its full-question two-ranker admission admitted none. This experiment sends the same candidates to a targeted evidence scorer and estimates the effect of the swap on the resulting list.

[xQuAD](https://theses.gla.ac.uk/4106/) assesses new documents against aspects represented in the selected list. [Joint passage ranking](https://aclanthology.org/2021.emnlp-main.560/) scores complementary passages together. Our [previous facet replay](TEMPORAL_CONFLICT_RERANKING_20261126.md#test-whether-a-candidate-adds-a-missing-answer) used category probabilities for a document's principal facet. The comparison here uses direct scores for both literal requirements per document and asks whether the actual retained documents can absorb a loss of evidence from the displaced item.

## Fixed protocol

Use the four frozen two-sided offers, current rank-ten document, and unchanged first nine for every query on the 191-topic panel. Reuse the low- and high-similarity literal requirements selected before relevance labels in the original search. This is a diagnostic reuse of the same topics. Rank only once per policy, with no threshold or model fitting. When the upstream method abstained, keep the original ranking in every arm, including the whole-question reference. Include all 191 questions in every result.

Score the gap and the represented requirement with the pinned Mixedbread checkpoint. Keep time wording as part of the original literal requirement. Use the same dispersed evidence selected from the complete question for every arm. Reuse the cached whole-question logits for the whole-question reference, verifying their candidate excerpts against the previous input cache. These relevance scores estimate support for a requirement.

For each of the two requirements, let the support of a list be the highest document score. For a candidate replacing rank ten, compute the new support minus the old support. The list-conditioned arm maximizes the change for the gap minus any drop in support for the represented requirement. Existing evidence for either requirement can come from any of the first nine documents. The matched pair arm applies the same gain-minus-loss formula to only the two exchanged documents, using exactly the same logits. A gap-only control ignores the second requirement. A whole-question control compares the two documents on the complete original query. Each arm applies at most one edit at position ten and keeps the original list unless its best change is strictly positive.

Freeze the rankings before reading train qrels and step judgments. Report final domain-macro nDCG@10, affected queries, steps restored and lost, and paired effects against both the original list and the matched pair control. A follow-up on the other two previously archived panels is warranted only if the list-conditioned rule beats both of these references in the macro and in every domain omission. Official dev and test labels are outside this experiment.

The 144 active questions produce 4,032 focus–document inputs. Every selected query view fits within the 192-token cap. Median lengths are 22 tokens for the gap and 30 for the represented request. A four-digit year-like token is present in 19 gap views and 21 represented views. The probe uses years and temporal wording as text, with no separate calendar classifier.

## Main results

| Rule on the 191 | Domain-macro nDCG@10 | Exchanges | Restored steps | Lost steps |
| --- | ---: | ---: | ---: | ---: |
| Maintained rank | 0.28212 | 0 | 0 | 0 |
| Whole-question score | 0.28134 | 77 | 2 | 1 |
| Gap only | 0.28251 | 59 | 2 | 1 |
| Pair benefit and loss | 0.28202 | 51 | 1 | 1 |
| List-conditioned benefit and loss | 0.28212 | 3 | 0 | 0 |

The matched list-minus-pair difference is +0.00010 [−0.00169, +0.00199], with a minimum domain-omission effect of −0.00064. Conditioning on the retained evidence almost eliminates the exchanges and reproduces the maintained per-query score. It fails the registered transfer screen. [Scorer cache and paired rankings](../results/analysis/list_conditioned_admission_191_20261128.json.gz)

On 59 active questions at least one offer beats the incumbent on the inferred gap alone. The highest score among the first nine absorbs that advantage on 56 of them. On the eight active questions with a judged-positive offer, none of those positives scores above the best original document on the inferred gap. An independent replay verifies every query score, domain mean, and step-count change.

## Privileged focus diagnostic

The retrieval archive already identifies eight uncovered annotated steps for which an offer exists. For those steps, replace the inferred gap with the original step instruction and score the same fourteen excerpts using the same checkpoint. The known supporting IDs determine the subset and hits. The model receives the step instruction and one excerpt. Count how often the top offer supports the step, beats the original tenth result, and scores above every original result. The last quantity tests whether the existing list would absorb the change under the maximum-support rule. This is a mechanism check using annotations and does not estimate deployment performance.

| Focus on eight uncovered steps across seven questions | Top offer supports the step | Supporting winner beats rank ten | Supporting winner beats every original result |
| --- | ---: | ---: | ---: |
| Inferred gap | 4 | 2 | 0 |
| Annotated step instruction | 2 | 2 | 1 |

The two views succeed on different steps when compared with rank ten. On `history/65653_514`, the original gap asks about federal law before admission to the Union. Naming Hawaii and its territorial governance before statehood in 1959 switches the winner to the judged supporting offer. Its excerpt describes annexation in 1898 and the creation of the territory in 1900. The highest-scoring original result on this step describes admission in 1959. Both excerpts mention 1959. In `history/65783_506`, the supporting offer for the wartime Norway step has no literal 1942 mention in the complete document. These cases motivate checking the event linked to a date and retaining a distinct state for unresolved time scope.

An independent replay verified all eight step labels, candidate identities, and relative score comparisons. Supplying the annotated instruction can fix the focus for an individual query. The overall counts point to evidence judgment as a separate bottleneck once focus is supplied. [Step diagnostics and cached scores](../results/analysis/focused_step_diagnostic_20261128.json.gz)

## Reproduction

Rebuild the input cache of the previous [admission experiment](CONTRASTIVE_EVIDENCE_RETRIEVAL_20261128.md#replay). Pass it as `--previous-inputs` to `replay_list_conditioned_admission.py prepare`, together with the archived 191-query candidates and the evaluated admission report as `--candidates` and `--admission`. Use `eval_fifth_ranker.py score` with the pinned Mixedbread model and batch size 16. Pass that output as `--focused` to `replay_list_conditioned_admission.py score`, retaining the same candidates and admission paths.

Run `diagnose_focused_step_scoring.py prepare` with those 191-query candidates, training data root, and the prepared focused inputs as `--inputs`. Score the privileged inputs with the same checkpoint. Use `diagnose_focused_step_scoring.py score` with the privileged scores and the original focused scores to recompute the diagnostic.
