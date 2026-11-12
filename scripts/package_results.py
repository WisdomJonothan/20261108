"""Build the Phase-0 results archive.

Includes every artefact that is cheap to ship and sufficient to re-score the
runs offline: TREC run files, per-query score JSONs, run metadata, logs, all
scripts (including the verbatim official starter kit), the environment
snapshot, the small query-side data files (qrels / steps / examples /
guidance) and a checksum manifest. The corpora, the Lucene indexes, the model
weights and the cached embeddings are NOT included (they are large and can be
re-downloaded or rebuilt).
"""

from __future__ import annotations

import hashlib
import os
import sys
import zipfile
from datetime import datetime, timezone

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIST = os.path.join(ROOT, "dist")
DOMAINS = ("iota", "bitcoin")
SMALL_DATA = ("qrels_dev.txt", "qrels_train.txt", "qrels_steps_dev.txt",
              "qrels_steps_train.txt", "steps_dev.jsonl", "steps_train.jsonl",
              "examples_dev.jsonl", "examples_train.jsonl",
              "guidance_dev.jsonl", "guidance_train.jsonl")


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def collect() -> list[tuple[str, str]]:
    """(archive_name, absolute_path) for every file to ship."""
    items: list[tuple[str, str]] = []

    def add_dir(sub: str, suffix: tuple[str, ...],
                skip_names: frozenset[str] = frozenset()) -> None:
        base = os.path.join(ROOT, sub)
        if not os.path.isdir(base):
            return
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in sorted(filenames):
                if name in skip_names:
                    continue
                if name.endswith(suffix):
                    items.append((rel(os.path.join(dirpath, name)),
                                  os.path.join(dirpath, name)))

    items.append(("PHASE0.md", os.path.join(ROOT, "PHASE0.md")))
    add_dir("results", (".md", ".json", ".jsonl"),
            frozenset({"tokenizer.json", "special_tokens_map.json",
                       "added_tokens.json", "tokenizer_config.json"}))
    add_dir("runs", (".trec",))
    add_dir("logs", (".log",))
    add_dir("scripts", (".py", ".sh", ".md", ".txt"))
    for dom in DOMAINS:
        for name in SMALL_DATA:
            p = os.path.join(ROOT, "data", "track1_tempo", dom, name)
            if os.path.isfile(p):
                items.append((f"data_meta/{dom}/{name}", p))
    def keep(a: str) -> bool:
        # fused views and train-split TRECs are derivable from the shipped
        # component runs; shipping them would add ~1 GB for no new evidence.
        if "/fused_" in "/" + a or a.split("/")[-1].endswith("_train.trec"):
            return False
        return True

    return [(a, p) for a, p in items if os.path.isfile(p) and keep(a)]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


README = """# RETECO Track 1a — Phase 0 results package

Produced {when} on the machine described in `results/env.json`.

## What is inside

| path | content |
|---|---|
| `results/PHASE0_REPORT.md` | the readable report: BM25 comparison, dense, union, temporal feasibility, latency, official-scorer cross-check |
| `results/score_*.json` | per-run metrics, including every per-query value |
| `results/*.meta.json` | run configuration and provenance (k1/b, model, batch size, timings, versions) |
| `results/official_scorer_*.json` | output of the organizers' `scorer.py` on the same run files |
| `results/union_*.json` | candidate-union recall / oracle analysis |
| `results/time_diag_*.json` | content-time signal counts |
| `results/env.json` | Python/Java/packages/GPU plus sha256 of the downloaded data |
| `runs/*.trec` | the 12 raw TREC run files (3 systems x 2 domains x 2 splits) |
| `logs/*.log` | retrieval and scoring logs |
| `scripts/` | all experiment code; `scripts/official/` is the organizers' starter kit, copied verbatim |
| `data_meta/` | qrels, steps, examples and guidance (small, query-side only) |

Not included, by design: `documents.jsonl` corpora, Lucene indexes, BGE weights,
cached embeddings, and the JDK. All are downloadable with the commands in
`PHASE0.md`.

## How to re-score without the corpora

```bash
export PYTHONUTF8=1        # required: scorer.py uses the platform default encoding
python scripts/score_run.py --run runs/bm25_pyserini_iota_dev.trec \
  --data <path-to-official-iota-dir> --split dev --json /tmp/check.json
python scripts/official/scorer.py --track 1 \
  --run runs/bm25_pyserini_iota_dev.trec \
  --qrels data_meta/iota/qrels_dev.txt --steps data_meta/iota/steps_dev.jsonl --json
```

`score_run.py` needs the official domain directory only for `steps_*.jsonl`
(period definitions); point it at `data_meta/<domain>` after copying the files
into the expected layout, or pass the official data directory directly.

## Headline dev numbers (nDCG@10)

| domain | topics | official BM25 | prebuilt-index BM25 | BGE zero-shot |
|---|---|---|---|---|
| IOTA | 3 | 0.3309 | 0.3852 | 0.4712 |
| Bitcoin | 30 | 0.0370 | 0.0300 | 0.2171 |

## Integrity

`MANIFEST.sha256` lists size and SHA-256 for every file in this archive.
"""


def main() -> int:
    os.makedirs(DIST, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = os.path.join(DIST, f"RETECO_Phase0_results_{stamp}_v16.zip")

    items = collect()
    manifest: list[str] = []
    for name, path in items:
        manifest.append(f"{sha256(path)}  {os.path.getsize(path):>12}  {name}")

    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("README_PACKAGE.md", README.format(when=when), zipfile.ZIP_DEFLATED)
        for name, path in items:
            z.write(path, name, zipfile.ZIP_DEFLATED)
        z.writestr("MANIFEST.sha256",
                   "\n".join(manifest) + "\n", zipfile.ZIP_DEFLATED)

    size = os.path.getsize(out)
    print(f"{out}\n{len(items)} files, {size / 1e6:.2f} MB")
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        print("zip test:", "OK" if bad is None else f"CORRUPT at {bad}")
        print("entries:", len(z.namelist()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
