# Submission pipeline

`scripts/make_submission.py` turns any query file into a submission-ready TREC
run with the configuration this project settled on:

```
mxbai-embed-large-v1 dense top-1000
  -> bge-reranker-v2-m3 on the dense top-100 (title query, 250-token window, max pooling)
  -> RRF(dense, CE), c = 60, dense:CE = 1:2   (weight chosen on train)
```

It takes `--queries <examples_*.jsonl>` rather than a split name, so the same
path is used for the dev sanity check and, later, for the hidden test queries.

```bash
python scripts/make_submission.py --domain iota \
  --queries data/track1_tempo/iota/examples_dev.jsonl \
  --index-data data/track1_tempo/iota \
  --out submissions/iota.trec

python scripts/official/format_checker.py submissions/iota.trec \
  --qrels data/track1_tempo/iota/qrels_dev.txt \
  --corpus data/track1_tempo/iota/documents.jsonl
```

## End-to-end check (IOTA dev, 2026-11-08)

* output: 3 topics × 1000 rows, tag `reteco-mxbai-ce-rrf`;
* `format_checker.py`: `lines: 3000 topics: 3 errors: 0 warnings: 0 — VALID`;
* `score_run.py`: nDCG@10 **0.8085**, Recall@100 0.9167 — identical to the
  offline `analyze_mxbai.py` number for RRF 1:2 on IOTA, so the pipeline is
  consistent with the evaluated system.

Runtime on IOTA (9,680 docs): dense encode ~2 min on the RTX 4060, CE rerank of
3 queries a few seconds. For the largest domain (travel, 158k docs) dense
encoding is ~1 h at ~46 doc/s; reranking adds ~2 s/query.
