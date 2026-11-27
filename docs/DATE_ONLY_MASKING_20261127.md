# Change the date while preserving the excerpt

The [controlled date-assignment probe](TEMPORAL_DATE_BINDING_20261127.md) found that rerankers can use dates attached to the wrong event. The [natural-data excerpt experiment](EVENT_DATE_WINDOW_20261127.md) also showed that recentering the excerpt on a date can remove useful context. This follow-up changes one year mention in that exact lexical excerpt.

## Protocol fixed before scoring

Use the same 86 History questions and 141 questions from the 10 other domains, with their existing dense top 14 and lexical excerpt. Both cohorts have already been used for excerpt comparisons. Treat isolated four-digit strings from 1000 through 2029 as candidate years. Exclude strings touching a dash or slash, strings named as years in the query, and repeated values in the excerpt. Identify dated query sentences with the earlier expression filter.

An excerpt is eligible when it has a candidate year with two or more terms from a dated query sentence within ten words in the same excerpt sentence, and a second candidate year in a different sentence with zero of those terms anywhere in its sentence. Choose the strongest linked candidate by lexical overlap, then its nearest eligible low-overlap candidate. The rule uses the query and excerpt. It does not read relevance labels or compare the numerical years to the cutoff.

Compare the original excerpt, one low-overlap year replaced by `[YEAR]`, and one linked year replaced by the same placeholder. Both treatments change four digits and preserve all other characters. When the rule abstains, all three arms share the archived score. Rescore the three versions together on changed documents using the same Qwen3-Reranker-0.6B prompt. Compare the standalone rank and the equal-weight z-normalized dense fusion. The main comparisons are low-overlap masking versus original and versus linked-year masking. Report coverage and paired estimates separately for History and the other domains, including questions with no eligible documents. Cross-domain improvement over both controls is the screening criterion. No official dev or test labels enter this experiment.

## Results

| Cohort | Editable excerpts | Fused original | Fused low-overlap mask | Fused linked-year mask |
| --- | ---: | ---: | ---: | ---: |
| History, 86 questions | 11 / 1,204 | 0.29589 | 0.29589 | 0.29589 |
| Ten other domains, 141 questions | 34 / 1,974 | 0.21412 | 0.21424 | 0.21412 |

The rule applies to **45 of 3,178 candidate excerpts** on 29 of 227 questions. One eligible excerpt belongs to a document marked positive. On the other-domain cohort, both primary contrasts are **+0.00012** with a paired query-bootstrap interval of [0, +0.00035]. The standalone comparison against the original loses 0.00006. Both positive fused contrasts come from the same Travel question. The other nine domain means tie. Rescoring the editable original excerpts changes some raw logits and reproduces the archived fused nDCG for every question.

In the sole fused improvement (`travel/185672_1170`), the edited document moves from rank 8 to rank 7. The heuristic masks 2017, which is when the passage says the relevant travel ban began. The linked-year control masks 2025, the reference date of the passage. Both are meaningful to the question. In another example, the heuristic labels 1000 transactions per block as a year. Checking 11 score-blind examples also finds cross-sentence coreference, dates of citations, and passages about different people with similar record types. The lexical rule measures local overlap. These cases show why identifying the actual event and the role of its date requires a distinct evaluation before applying date masking in the ranking pipeline.

The small shift leaves the official reference in place. An independent check reconstructed all 45 edited passages, verified their digests and mask positions, and replayed all 227 queries from the frozen scores. The 135 unique reranker inputs took 10.6 seconds of inference.

## Replay

```text
python scripts/eval_date_only_masking.py score --data-root data/track1_tempo --archives results/analysis/event_date_window_history_20261127.json.gz results/analysis/event_date_window_transfer_20261127.json.gz --model models/qwen3-reranker-0.6b --output mask-scores.json.gz
python scripts/eval_date_only_masking.py evaluate --data-root data/track1_tempo --scores mask-scores.json.gz --output mask-evaluated.json.gz
```

The evaluated archive also retains the input scores for replay. The masks have character offsets within the original excerpt. Its start is the whitespace word offset after HTML normalization, and its digest verifies exact reconstruction.

[Mask positions, paired scores, and per-query evaluation](../results/analysis/date_only_masking_20261127.json.gz)
