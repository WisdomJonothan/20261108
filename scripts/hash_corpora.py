#!/usr/bin/env python3
"""Record a sha256 + row count for every domain corpus and task file.

Why: `run_chunk_rerank.py` resolves the corpus as
`data/track1_tempo/<domain>/documents.jsonl` regardless of which run file it is
re-ranking. If the organizers re-release the corpus (v1.0 -> v1.1 already
happened once), an old run would be silently scored against a new corpus and the
mistake would look like a modelling result. Record the fingerprints once, put
them in the submission package, and diff them at submission time.

Usage:
  python hash_corpora.py                 # write results/corpus_manifest.json
  python hash_corpora.py --check         # re-hash and fail on any difference
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "track1_tempo")
MANIFEST = os.path.join(ROOT, "results", "corpus_manifest.json")
TASK_FILES = ["examples_train.jsonl", "examples_dev.jsonl",
              "qrels_train.txt", "qrels_dev.txt"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def build():
    out = {"domains": {}, "totals": {}}
    total_docs = 0
    for d in sorted(os.listdir(DATA)):
        dd = os.path.join(DATA, d)
        if not os.path.isdir(dd):
            continue
        rows = {}
        cp = os.path.join(dd, "documents.jsonl")
        if os.path.exists(cp):
            rows["documents.jsonl"] = {"sha256": sha256(cp), "rows": count(cp),
                                       "bytes": os.path.getsize(cp)}
            total_docs += rows["documents.jsonl"]["rows"]
        for name in TASK_FILES:
            p = os.path.join(dd, name)
            if os.path.exists(p):
                rows[name] = {"sha256": sha256(p), "rows": count(p),
                              "bytes": os.path.getsize(p)}
        out["domains"][d] = rows
    out["totals"] = {"domains": len(out["domains"]), "documents": total_docs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    fresh = build()
    if args.check:
        if not os.path.exists(MANIFEST):
            print("no manifest to check against; run without --check first")
            return 2
        with open(MANIFEST, encoding="utf-8") as f:
            old = json.load(f)
        bad = []
        for d, rows in old["domains"].items():
            for name, meta in rows.items():
                now = fresh["domains"].get(d, {}).get(name)
                if not now or now["sha256"] != meta["sha256"]:
                    bad.append(f"{d}/{name}")
        if bad:
            print(f"MISMATCH in {len(bad)} file(s): {bad[:8]}")
            return 1
        print(f"all {sum(len(v) for v in old['domains'].values())} fingerprints match "
              f"({fresh['totals']['documents']} documents, {len(fresh['domains'])} domains)")
        return 0
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(fresh, f, indent=1)
    print(f"{fresh['totals']['domains']} domains, {fresh['totals']['documents']} documents")
    print(f"-> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
