# Annotation rubric — why does a gold document's year differ from the query's year?

You are coding **mechanism**, not relevance: every item is already known to be
gold (relevant).  The question is *why* a document whose text mentions years
outside the query's time window is still relevant.

You get, per line of `annotator_sheet.jsonl`:

- `item_id` — anonymous id; do **not** try to look the document up
- `query`, `query_summary`, `query_time_anchors`, `query_years_seen`
- `doc_years_seen` — years extracted from the document's first 1200 characters
- `doc_snippet` — the first ~700 characters of the document (HTML stripped)

Do not read `key.json`, `runs/`, `data/` or any qrels file: the sheet is
deliberately blinded so that the coding cannot use the document id, the domain
name or the retriever score.  Code from `query` + `doc_snippet` only; use
`doc_years_seen` / `query_years_seen` only to understand which years triggered
the item.

## Labels (choose exactly one)

| label | meaning |
|---|---|
| `retrospective` | The document recounts or analyses events **from an earlier period** (history article, post-mortem, anniversary, "how X began in 2015"). The query topic is still served, but the time it talks about is the past. |
| `continuing` | The document is about an **ongoing state or a timeless subject**; the extracted year is a start date, an established-since date, or a general reference. Any "time" in the query is satisfied by the subject still existing / still being current. |
| `incidental_year` | The year is **not the time of the events** — a citation, dataset name, version number, price example, table entry, legal reference, publication year of a source, etc. The document's content time is unconstrained or matches. |
| `period_shift` | The document addresses a **closely adjacent period** (annual/quarterly report, fiscal year, rolling window) whose year label is off by a small amount, but its content overlaps the query's window. |
| `wrong_time` | The document genuinely addresses **another period**, materially different from the query's window; its relevance comes from topic only, not from time. |
| `unclear` | The snippet is too short or too ambiguous to decide. |

## Fields per item

```json
{"item_id": "…", "mechanism": "retrospective", "evidence": "…", "confidence": 1}
```

- `evidence`: a verbatim quote of **at most 20 words** from `doc_snippet` that
  justifies the label.  Use "" only for `unclear`.
- `confidence`: 1 = weak, 2 = medium, 3 = strong.
- One JSON object per line, UTF-8, in the order of the items you were assigned.
- Write **only** to the output file you were given; never modify the sheet.

## Rules

1. Judge mechanism from the snippet, not from what a domain "usually" does.
2. If a document mentions several years and the snippet's own time is the
   query window, prefer `incidental_year`.
3. `continuing` requires a reason visible in the snippet (a present-tense
   description, a "since 2012" phrase, a how-to body), not just absence of
   narrative time.
4. Never infer the domain or the document id.  Guessing the topic is fine.
5. Be conservative: when two labels fit, pick the one the evidence sentence
   supports best and lower confidence.
