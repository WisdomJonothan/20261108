# Check which event a year belongs to before editing rank ten

The [date assignment probe](TEMPORAL_DATE_BINDING_20261127.md) shows that hiding a different event's date improves strict temporal preference on matched evidence pairs. The [list-conditioned admission replay](LIST_CONDITIONED_EVIDENCE_ADMISSION_20261128.md) shows that existing documents can suppress a candidate with a higher score than the incumbent. This experiment tests whether limiting that comparison to documents with identifiable event–year evidence helps. [Understand before Answer](https://aclanthology.org/2022.naacl-main.28/) separates the referenced event and the temporal relation for reading comprehension. Here we use a fixed rank-ten decision to measure the downstream cost and benefit of using a year check in a reranker.

## Protocol fixed before evaluation

Use the 186, 155, and 191 archived training panels. Each question gets its maintained top ten and the same four outside offers across all arms. The 191 reuse the two-sided offers. For the other two panels, alternate the archived forward and reverse offers, keeping the first four distinct IDs. These three panels have prior project exposure. No new parameters are fitted. Preserve the first nine positions in every arm.

Activate only if a single interrogative sentence yields a single supported year-scale expression. Interpret before and after strictly and between, from, by and since inclusively. The in and during forms check year equality. Extract no year from a range-like document token. Unrecognized expressions leave the ranking intact. Take the same dispersed excerpt selected by the complete original question for every arm. Use the frozen Mixedbread scorer on the literal dated question sentence and the excerpt. Use the same scalar scores everywhere.

The automatic year link requires at least two distinct overlap terms from the dated question after removing years and common question and temporal terms. Find the sentences with the highest overlap. When they all have a single unique year and agree, check it against the query's bound. Record the supporting quotation and the year. Treat other cases as unresolved. This is a sentence-level proxy for event grounding. It can misidentify the event and can miss paraphrases, references and implicit dates. For range questions the year check only verifies compatibility of observed evidence and the interval.

Every challenger must have a strictly higher relevance score than rank ten. The focused control uses only that condition. The score-head control also requires a strict win over all first-nine scores. The any-year control allows only challengers with any compatible year and uses only first-nine documents that pass that same condition as additional score floors. The linked-pair arm requires a linked compatible year for the challenger. The linked-head arm also uses the scores of any first-nine documents that have a linked compatible year. Each rule selects its highest-scoring qualifying challenger and leaves rank ten intact if none qualifies. An unresolved or conflicting year does not make an incumbent ineligible to stay.

Freeze and hash the rankings before opening training qrels or step labels. Report panel and pooled domain-macro nDCG@10, rank-ten changes, affected queries, and annotated steps restored or lost. Advance the linked-head policy only if it improves over the maintained ranking, focused score and any-year control in each panel, the pooled analysis, and each domain-omission check. This comparison uses no official dev or test judgments.

## Source-pair diagnostic protocol

Replay the two deterministic year checks on the eleven previously reviewed [source pairs](TEMPORAL_QUERY_CONTRAST_20261125.md), using the same authentic passages and two question variants. Count a choice only when exactly one passage passes the check. Record selection of the expected passage, selection of the other passage, or abstention, and count a flip only when both variants select the expected passage. The author's event-time annotations supply the expected choices and are hidden from the checks. They are used to test the extractor separately from the relevance scorer.

## Results

The query parser triggers on 16 of the 532 questions, including three on the 186 panel, seven on the 155 and six on the 191. The 224 query–document pairs take 13.5 seconds of inference on the local RTX 4060 Laptop GPU. All dated questions fit in 38 model tokens and the longest full pair is 456 tokens. The evidence excerpts are therefore inside the model input budget for this replay.

| Rank-ten rule | 13-domain macro nDCG@10 | Changes | Improved / declined topics | Restored / lost steps |
| --- | ---: | ---: | ---: | ---: |
| Maintained ranking | 0.31878 | 0 | 0 / 0 | 0 / 0 |
| Focused score | 0.31886 | 8 | 1 / 0 | 1 / 0 |
| Focused score with first-nine maximum | 0.31878 | 1 | 0 / 0 | 0 / 0 |
| Any compatible year | 0.31878 | 1 | 0 / 0 | 0 / 0 |
| Linked year, pair only | 0.31878 | 0 | 0 / 0 | 0 / 0 |
| Linked year, conditioned on the first nine | 0.31878 | 0 | 0 / 0 | 0 / 0 |

Of the 64 active offers, seven have a compatible year somewhere in the excerpt and none pass the same-sentence check. Four of the 160 incumbent excerpts pass that stricter check. The ranked list is unchanged under both linked rules, so the transfer gate is closed. The sole positive edit on the focused arm is Hawaii question `history/65653_514`. Its extracted time is the hypothetical law's 1934 date, while a retrieved supporting document describes territorial governance before statehood in 1959. This illustrates a query-side binding error that finer date handling inside the candidate alone leaves unresolved.

| Reviewed source-pair check | Correct / 22 | Wrong / 22 | Abstained / 22 | Correct flips / 11 |
| --- | ---: | ---: | ---: | ---: |
| Any compatible year | 10 | 0 | 12 | 3 |
| Linked compatible year | 2 | 1 | 19 | 0 |

In the Canadian bank-note pair, the link chooses an incidental 1817 date and misses the later phase-out being asked about. The source-pair result shows that restricting the check to a lexically selected sentence does not establish an event-to-year relationship on natural passages. The eleven pairs were selected for explicit dates and are a probe, with both variants of each case dependent on the same two passages. [Diagnostic pair decisions](../results/analysis/event_year_admission_source_pairs_20261128.json.gz)

## Where more scoring could help

With qrels, 24 of the 532 questions can improve if only rank ten is allowed to change among the fixed four offers. The ideal score is **0.32385** (+0.00507). Just three of those 24 questions activate the year check. Under the current trigger, perfect selection raises the score only to **0.31909** (+0.00030). On those same 532 questions, ideal reordering within the maintained ten reaches **0.46944** and ideal reordering of all fourteen reaches **0.48294**. These label-assisted ceilings quantify the listed actions on the fixed candidates. They favor testing evidence at earlier ranks and on broader candidates over further tuning of the rank-ten year gate.

Independent arithmetic verified all scores, top-nine preservation, exchanges, and step changes against the official training files. Official dev and test were outside the replay. [Fixed rankings, logits, witness hashes and oracle results](../results/analysis/event_year_admission_532_20261128.json.gz)

## Replay

Run from the repository root with the original training files. Use fresh output paths.

```text
python scripts/replay_event_year_admission.py prepare --candidates results/analysis/contrastive_evidence_retrieval_341_20261128.json.gz results/analysis/contrastive_evidence_retrieval_191_20261128.json.gz --archive-dir results/analysis --data-root data/track1_tempo --output fixed.json.gz --score-inputs inputs.json.gz
python scripts/eval_fifth_ranker.py score --inputs inputs.json.gz --model models/mxbai-rerank-large-v1-98f6558 --batch-size 16 --output scores.json.gz
python scripts/replay_event_year_admission.py evaluate --data-root data/track1_tempo --fixed fixed.json.gz --score-inputs inputs.json.gz --scores scores.json.gz --output results.json.gz
python scripts/replay_event_year_admission.py diagnostic --manifest results/analysis/temporal_minimal_pairs_manifest_20261125.json --data-root data/track1_tempo --output source-pairs.json.gz
```
