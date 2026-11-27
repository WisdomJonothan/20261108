# Select an excerpt by the event near its date

The [matched date-assignment probe](TEMPORAL_DATE_BINDING_20261127.md) found that hiding unrelated dates helps both rerankers on short controlled texts. This follow-up uses original RETECO training questions, passages, and relevance judgments.

[SERank](https://doi.org/10.1016/j.eswa.2026.134288) uses query-aware sentence filtering before reranking. Here the matched comparison measures the extra value of linking local topical terms to the date anchor within a shared excerpt bank.

## Protocol fixed before scoring

Use training questions outside the union of the preceding 641- and 242-query replay panels and the 11 date-assignment source question IDs. Require a sentence with an explicit year following a temporal expression. Fix the first 14 original mxbai documents for each question, before reading any judgments. The ID inventory showed that all 328 unused questions in the preceding panels' complement belong to History. Before opening any labels, expand the initial 12-question smoke test to every eligible History question. The outcome is a History-domain comparison. The smoke test has no separate relevance evaluation.

Compare three excerpt policies. All use the same bank of windows whose length is 96 whitespace words or the whole document if shorter. The bank includes the overlap-strided baseline and windows centered around individual four-digit year mentions. The lexical excerpt matches the complete question's non-year terms. The date-only excerpt chooses the best lexical match among year-centered windows. The linked-date excerpt picks a year near two or more terms from a dated question sentence. It maximizes their count within ten words of the date in the same document sentence. Both date-aware arms fall back to the lexical excerpt when no linked window is available. Year values alone carry no selection weight. Dates are compared by Qwen3-Reranker-0.6B, with its default instruction, the same complete question and one excerpt for every document in every arm. Identical prompts share one prediction across the arms.

Report standalone and fixed equal-weight z-normalized dense fusion, with fused linked-date versus fused lexical as the primary contrast. The fused date-only arm is the matched control for the link criterion. Save the raw scores and excerpt offsets before opening the training judgments. Report History nDCG@10, paired query-bootstrap intervals and directional changes. Positive primary and date-only contrasts would motivate a cross-domain test. Official dev and test stay outside the experiment. This screening experiment uses the baseline dense candidate pool and a local reranker.

## History holdout results

| Excerpt selection | Standalone nDCG@10 | With fixed dense fusion |
| --- | ---: | ---: |
| Dense alone | 0.29035 | |
| Lexical | 0.29312 | 0.29589 |
| Date only | 0.29998 | 0.29706 |
| Linked date | 0.30085 | 0.29833 |

The linked excerpt gains +0.00244 [−0.00597, +0.01152] over lexical with fusion and +0.00127 [−0.00583, +0.00932] over date-only. These are paired 95% query-bootstrap intervals. Against lexical, 11 questions improve, six decline, and 69 tie. Omitting the most improved question changes the macro estimate to −0.00029. The selector changes 157 of 1,204 excerpts across 51 of the 86 queries. It changes a judged relevant document's excerpt on 22 queries. Dense top 14 contains a judged positive on 63 queries. The raw scores, overlap check and official-training nDCG calculations were independently verified.

On the question about burial arrangements before 1880 (`70556_274`), a different excerpt lowers the first-ranked candidate's score. Both relevant documents keep their excerpts and move from positions 2–3 to 1–2. On the Eastern Front question (`58889_653`), the linked date window drops the lead-in that connects the passage to the broader question. The relevant document falls from rank 1 to rank 2. These examples were inspected after scoring. A year can mark the question's requested period or describe background that a broader answer must put in context.

## Frozen cross-domain transfer

The two primary History point estimates are positive. Apply the same rule, 96-word excerpts, checkpoint, dense depth and fusion weight to every other-domain question from the archived panels meeting the same explicit-year condition. Exclude the earlier source-case IDs. This extends the test to different domains and uses previously studied training questions. Compare the same two fused contrasts and the minimum change after leaving out any single domain. These scores describe the covered subset of train domains.

| Excerpt selection on 141 questions in 10 domains | Standalone domain-macro nDCG@10 | With fixed dense fusion |
| --- | ---: | ---: |
| Dense alone | 0.20750 | |
| Lexical | 0.20496 | 0.21412 |
| Date only | 0.20294 | 0.21203 |
| Linked date | 0.20342 | 0.21139 |

The primary contrast reverses to **−0.00273** [−0.00621, +0.00022] with query resampling within domain, and [−0.00591, −0.00029] with domain resampling. Seven questions improve and nine decline. Four domain means decline and six tie. The primary point estimate stays negative after omitting any one domain. The matched contrast against date-only is −0.00064 [−0.00417, +0.00322] with query resampling. The fixed rule fails transfer on this sample and remains outside the official pipeline. Cardano and IOTA have no eligible queries under the expression filter.

The controlled masking probe kept the surrounding evidence. The natural-data rule sometimes replaces that evidence while pursuing a date. A more direct follow-up would preserve the lexical excerpt and evaluate date attachment only when the model can point to the event and its date in the same passage. A separate annotation of whether the date belongs to the question's actual request would distinguish missing temporal evidence from a date mentioned in background text.

Independent replay verified every query's population, all original dense candidate IDs and scores, equal scoring for identical excerpts, and the per-query nDCG. Qwen inference took 87.5 seconds for 1,396 distinct History prompts and 128.6 seconds for 2,282 distinct transfer prompts on the local RTX 4060 Laptop GPU.

## Replay

```text
python scripts/eval_event_date_windows.py score --data-root data/track1_tempo --runs runs --exclude-archive results/analysis/paired-membership-20261127.tar.gz --exclude-source-cases results/analysis/temporal_minimal_pairs_manifest_20261125.json --model models/qwen3-reranker-0.6b --output history-scores.json.gz
python scripts/eval_event_date_windows.py evaluate --data-root data/track1_tempo --scores history-scores.json.gz --output history-evaluated.json.gz
```

Add `--transfer` to the scoring command and use new filenames for the other-domain comparison. The evaluated archives retain the raw score payload, so either archive can also be passed directly to the second command to recompute its evaluation.

[History query IDs, excerpt offsets, raw scores, and evaluation](../results/analysis/event_date_window_history_20261127.json.gz) · [Transfer data and evaluation](../results/analysis/event_date_window_transfer_20261127.json.gz)
