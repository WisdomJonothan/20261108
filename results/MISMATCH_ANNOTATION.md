# Do year-mismatched gold documents mean the system missed the time condition?

Status: dev only (v1.1, 519 Track 1a dev queries). No test data was touched.
Written 2026-11-09 for `results/NEXT_STEPS.md` P2 #5.

Short answer: **no.** Of 35 blind-coded gold documents whose text carries a year
outside the query's time span, **one** genuinely addresses a different period.
The other 34 are retrospective accounts, incidental year mentions, ongoing
subjects, or adjacent-period documents — cases where demoting the document for
being "time-inconsistent" would remove a *relevant* document.

## 1. How much time evidence do gold documents carry at all?

`scripts/gold_year_anatomy.py` (`results/gold_year_anatomy.json`). Document time
is the high-confidence extraction of `hardneg_mine.doc_time`: a four-digit year in
the title or first 1,200 characters, kept when there is one year, or the span is
≤ 3 years, or one year dominates with span ≤ 10.

| domain | gold | with year | share | mismatch (strict def) |
|---|---:|---:|---:|---:|
| bitcoin | 94 | 17 | 18.1% | 3 |
| cardano | 43 | 3 | 7.0% | 0 |
| economics | 95 | 38 | 40.0% | 4 |
| genealogy | 94 | 23 | 24.5% | 6 |
| history | 1,010 | 283 | 28.0% | 58 |
| hsm | 115 | 13 | 11.3% | 4 |
| iota | 9 | 3 | 33.3% | 0 |
| law | 29 | 10 | 34.5% | 0 |
| monero | 47 | 10 | 21.3% | 0 |
| politics | 124 | 55 | 44.4% | 14 |
| quant | 21 | 1 | 4.8% | 0 |
| travel | 73 | 14 | 19.2% | 1 |
| workplace | 29 | 5 | 17.2% | 0 |
| **total** | **1,783** | **475** | **26.6%** | **90** |

Two readings matter:

* **Only a quarter of the gold documents expose a year.** A perfect year filter
  can therefore speak about at most ~27% of the relevance mass; for the rest the
  time condition has to be satisfied through text semantics, which no simple
  rule captures.
* Among gold documents that *do* expose a year, a large share disagrees with the
  query span. The exact share depends on the consistency definition, so we
  measured both (next section).

## 2. Two definitions of "consistent" (and why the deployed ceiling over-counts)

`hardneg_mine.query_interval` keeps only the **largest** year a query mentions
unless the text contains an explicit `2015-2017` range. For a query that lists
several anchors (`1945, 1946, ..., 1975`) a document about 1946 is then labelled
mismatched. `scripts/check_mismatch_definitions.py` compares:

* **def1 (max-year interval, the one used by `temporal_ceiling.py`)**:
  120 / 246 year-bearing golds are mismatched (**48.8%**).
* **def2 (anchor span)**: consistent iff a document year falls within
  `[min(query years) − 1, max(query years) + 1]`; mismatched **73 / 246
  (29.7%)**.

47 pairs change side. That means the ceiling analysis reported in
`results/TEMPORAL_CEILING.md` over-counts time mismatches among *gold* documents,
and therefore over-estimates what a year-based filter could win, by roughly 20
percentage points of the year-bearing population. The ceiling conclusion does
not flip — demotion already hurt under def1 — but the number should be quoted
with the definition.

Restricting to the 632 gold documents of queries where the interval exists:
246 (38.9%) carry a year, 73 of those (29.7%) are outside the anchor span.

## 3. Blinded annotation

Frame: dev, frozen dense `mxbai-embed-large-v1` top-1000, gold documents with an
extractable year outside the anchor span. 107 such pairs exist; the sampler
(`scripts/annotate_mismatch_sample.py`, top-1000, cap 20/domain) drew 53 items
stratified by domain.

Blinding: the coding sheet stripped the domain, the docid and every retriever
score; each item shows the query, its guidance summary and anchors, the
extracted years and the first ~700 characters of the document. Coders were told
not to open `key.json`, `data/`, `runs/` or any qrels.

Rubric: `results/mismatch/ANNOTATION_RUBRIC.md`, six labels —
`retrospective`, `continuing`, `incidental_year`, `period_shift`, `wrong_time`,
`unclear` — plus a verbatim evidence quote (≤ 20 words) and a confidence 1–3.

Reliability: three coders covered disjoint ranges; a fourth re-coded the first
20 items independently. **Cohen's κ = 0.84** (90% raw agreement, 2 disagreements
over 20 items), which the code reports per item in
`results/mismatch/summary.json`.

Of the 53 sampled items, 18 are artefacts of the max-year heuristic (the query
names the document's year, so def2 counts them consistent). They are removed from
the headline (`results/mismatch/def2_filter.json`), leaving **35** items:
history 11, politics 10, genealogy 5, economics 3, hsm 3, bitcoin 2, travel 1.

## 4. Mechanism distribution (35 strict items)

| label | n | share | what it looks like |
|---|---:|---:|---|
| `retrospective` | 13 | 37.1% | history/genealogy articles that recount earlier events; the query's "current state" is answered through its past |
| `incidental_year` | 11 | 31.4% | the year is a citation, dataset name, version label, statute name or example price — not the time of the events |
| `continuing` | 8 | 22.9% | ongoing subject (still-standing law, still-used mechanism, resource page); the year is a start/introduction date |
| `period_shift` | 2 | 5.7% | adjacent annual/administrative period (report year vs coverage year) |
| `wrong_time` | 1 | 2.9% | the document truly discusses another period and is relevant on topic only |
| `unclear` | 0 | 0.0% | — |

Examples (from the coded evidence quotes): a 2018 Dodd-Frank rollback document
answering a 2007–08 crisis query (`continuing`), a bibliographic note where 1995
is a citation year (`incidental_year`), an Ohio death-records resource whose
1867/1908 history is administrative context (`retrospective`), and a tariff
argument citing the 2018–19 episode (`period_shift`).

**Only 1 of 35 year-mismatched gold documents is genuinely about another
period.** The dominant pattern is that the year in the text is not the time of
the events — exactly the query-time / publication-time / content-time
distinction the task is about, and exactly what a year-consistency feature
conflates.

## 5. Why this matters for the system and the paper

* The earlier null results on temporal reranking (`results/TEMPORAL_GATE.md`,
  `results/TEMPORAL_CEILING.md`) are not a failure to find the right feature
  weight; they follow from what the evidence is. A year-consistency score is
  wrong on ~97% of the mismatched gold documents, so any monotone use of it
  trades relevant documents for a time condition the annotators themselves did
  not apply.
* It also explains why the ceiling experiment — sort consistent-first — lost
  nDCG on 7 of 8 anchored domains: the "inconsistent" block is mostly relevant.
* The remaining temporal lever would have to extract *event* time from prose
  (what the document is about, when that happened) rather than any year string.
  Our attempts at this on 8 GB (`s_temp`, the CE temporal statement) move within
  noise, consistent with this annotation: the signal is there (`s_temp` AUC up to
  .88) but it is not what the retriever is missing.

Caveats: n = 35 strict items; history and politics supply 21 of them, so the
retrospective share is inflated by the corpora where long-form history writing
dominates; single annotator per item for 15 of 35 (the other 20 were
double-coded). This is evidence for a mechanism, not a population estimate —
the population-level statement stays the observability table in §1.

## 6. Reproduce

```bash
python scripts/gold_year_anatomy.py                 # §1 table
python scripts/check_mismatch_definitions.py        # §2, also writes the strict filter
python scripts/annotate_mismatch_sample.py --k 1000 --per-domain 20
# coding: results/mismatch/part[ABCD]*.jsonl (blinded sheets + rubric)
python scripts/aggregate_mismatch.py                # §4 table + kappa
```

Files: `results/mismatch/{annotator_sheet.jsonl,key.json,part*.jsonl,summary.json,
def2_filter.json,ANNOTATION_RUBRIC.md}`, `results/gold_year_anatomy.json`,
`results/mismatch_definitions.json`.
