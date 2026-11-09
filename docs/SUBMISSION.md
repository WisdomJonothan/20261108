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

## End-to-end check

* IOTA dev (2026-11-08, zero-shot CE, RRF 1:2 — the retired config): 3 topics x
  1000 rows, official `format_checker.py` `errors: 0`, `score_run.py` nDCG@10
  0.8085. That run validated the plumbing, not the current weights.
* **Synthetic test replay (2026-11-10, GPU, law/quant/iota, 24 topics):**
  `scripts/phase13_test_replay.sh law quant iota` with
  `--model-ce results/ce_ft/v2m3_pool8_e2/best`, `--fusion z --wa 1 --wb 1`.
  Official checker on all three files: `errors: 0`, `RESULT: VALID`.
  Result: `PHASE 13: PASS`.
  The first GPU attempt failed the strict-run sanity on `quant/70097_1088`:
  ranks 48/49 and 51/52 printed identical scores. Two bugs in the writer:
  de-tie compared *raw* floats (two scores 1e-9 apart both round to the same
  six-decimal value) and the tail filler used `-1e9 + 1e-6*i`, which made the
  de-tie step `abs(prev)*1e-6 = 1000` and walked the filler score to
  `-1.0e9-899`. Both fixed in `make_submission.py`: de-tie now operates in
  printed space (step exactly 1e-6, never rounded away) and the filler is
  `-1e9 - 1e-3*i` (rank-spaced, no accumulation). The replay then passed for
  all three domains. Lesson: the official checker accepts equal adjacent
  scores; the strict-run check is what catches the ambiguity, so keep it.
* The current pipeline is `--fusion z --wa 1 --wb 1` with either the zero-shot
  reranker or `--model-ce results/ce_ft/v2m3_pool8_e2/best`. The CE cache file is
  keyed by the checkpoint directory name (`ce_<domain>_<model>.trec`), so a
  zero-shot cache can never be silently reused for a fine-tuned model.
* Dev-format runs of the frozen system live in `runs/sub_sft11_<domain>_dev.trec`
  (13/13 official-checked). Those are produced by the offline pipeline, not by
  this script; `scripts/test_mode_dryrun.sh` is the script that fabricates a test
  split from dev and exercises this path end to end.

## Test phase runbook (January 2027)

1. Drop the released `examples_test.jsonl` into `data/track1_tempo/<domain>/`.
2. Per domain:
   `python scripts/make_submission.py --domain <d> --queries data/track1_tempo/<d>/examples_test.jsonl --index-data data/track1_tempo/<d> --out submissions/<d>.trec --fusion z --wa 1 --wb 1 --model-ce results/ce_ft/v2m3_pool8_e2/best`
3. Validate every file with `scripts/validate_submission.py` against the test
   qrels if released (otherwise at least the topic-id set and the format checker
   without `--qrels`), then de-tie (`--detie`) — the de-tie is ordinal and
   re-validates its own output.
4. Package: `scripts/package_results.py` + `RUN_MANIFEST.sha256`.

Runtime on IOTA (9,680 docs): dense encode ~2 min on the RTX 4060, CE rerank of
3 queries a few seconds. For the largest domain (travel, 158k docs) dense
encoding is ~1 h at ~46 doc/s; reranking adds ~2 s/query at 250-token windows,
more for long documents (history averages ~30 s/query on dev).
