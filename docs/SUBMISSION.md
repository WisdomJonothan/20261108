# Submission pipeline

`scripts/make_submission.py` turns any query file into a submission-ready TREC
run with the configuration this project settled on:

```
mxbai-embed-large-v1 dense top-1000
  -> bge-reranker-v2-m3 on the dense top-100 (title query, 250-token window, max pooling)
  -> z-score fusion of dense and CE, weight 1:1
     (chosen on TRAIN under the official domain-macro statistic)
```

RRF is still available (`--fusion rrf`, `c = 60`, default `dense:CE = 1:2` via
`--rrf-wa/--rrf-wb`); the z weights are `--wa/--wb`. Both branches now take
their weights from flags instead of module constants, so a run's configuration
is always reproducible from its command line.

It takes `--queries <examples_*.jsonl>` rather than a split name, so the same
path is used for the dev sanity check and, later, for the hidden test queries.

## Cache provenance

Both intermediate runs are cached (`tmp/submission/dense_<domain>.trec` and
`ce_<domain>_<checkpoint>.trec`). Each cache file carries a fingerprint sidecar
(`*.fp.json`) over everything that changes its bytes:

* dense: sha1 of the query file, the index directory, the dense model, the
  pooling, the query prefix and `--topk`;
* CE: sha1 of the dense run, the checkpoint path, `q-mode/q-cap/win/stride/pool`
  and `--k`.

On a mismatch the script refuses to reuse the cache (`SystemExit`, with the two
fingerprints printed) instead of silently scoring a different query file;
`--refresh` forces a rebuild, `--strict-cache` also rejects legacy caches that
have no sidecar. A legacy cache without a sidecar is **adopted**: it is reused
once and stamped with the current fingerprint (`"provenance": "adopted-legacy"`),
so the next run does catch a changed query file — verified on the iota replay,
where a two-query query file is refused with
`[cache] dense mismatch: ... was built with adf15789aa03423c, current inputs give 2edb931a8d354821`.
The fusion configuration and both fingerprints are written to
`<out>.config.json` next to the run, so a submitted file can be traced back to
the exact inputs that produced it.

Rebuilding the iota submission from the legacy caches reproduces the frozen
file byte for byte (`md5 a210d92a9f06317cbb2a17adf22387a2`, 1.2 s, no GPU).

The document-embedding cache in `run_dense_generic.py` uses the same scheme
(`doc_emb_*.npy.fp.json`, over model/pooling/max-length/dtype/prefix/document
texts), which is what makes a zero-shot and a fine-tuned encoder impossible to
confuse.

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
  Result: `PHASE 13: PASS`. Independently re-checked (2026-11-10 04:55): iota
  3000 rows / 3 topics, law 11000 / 11, quant 10000 / 10, `errors: 0`,
  `warnings: 0` each, and a strict scan found **0** rank-order violations and
  **0** exact printed ties at 1000 rows per topic — i.e. the de-tie fix is in
  effect in the files that were kept.
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
  keyed by the checkpoint directory name (`ce_<domain>_<model>.trec`) *and*
  fingerprinted over the dense run and the rerank parameters, so neither a
  zero-shot cache nor a cache built for another query file can be silently
  reused for a fine-tuned model.
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

## Packaging all test domains (January)

```bash
DOMAINS="iota law quant" bash ops/submit_all.sh /path/to/test_queries     results/ce_ft/v2m3_pool8_e2/best results/submission_pkg
```

`ops/submit_all.sh` builds one submission per `examples_test_<domain>.jsonl` it
finds, runs the official `format_checker.py` **and** the local
`scripts/check_submission_strict.py` (contiguous ranks, strictly decreasing
scores, no missing topics), then writes `MANIFEST.json` with each file's md5,
line count and the full provenance block from `<out>.config.json` (query file
sha1, fusion weights, CE q-mode/cap/window/pool, dense and CE fingerprints).

* `WORK_ROOT=<dir>` reuses an existing cache directory (a dry run against the
  synthetic replay, `WORK_ROOT=results/test_replay`, reproduces the frozen iota
  file byte for byte: `md5 a210d92a9f06317cbb2a17adf22387a2`, 3000 lines).
* The script exports `PYTHONUTF8=1` itself: the official checker opens files
  with the locale codec and dies on UTF-8 corpora on a Chinese Windows.
* Exit code is non-zero if any domain fails either check; a failed domain is
  reported and the rest still build.
