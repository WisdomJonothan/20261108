#!/usr/bin/env python3
"""Fetch the Track-1a files that are missing locally from the HF mirror.

The local release was assembled from a partial download: five domains
(cardano, law, quant, travel, workplace) have no *_train files although the
official release ships them (official split is 70/30 per domain, 1,211 train
queries). This script lists every domain directory on the mirror, compares it
with `data/track1_tempo/<domain>/`, and downloads only the missing files,
verifying size and JSONL/QRELS parseability afterwards.

Usage:
  python fetch_missing_train.py [--domains law cardano ...] [--dry-run]
"""
import argparse
import json
import os
import sys

import requests

BASE = "https://hf-mirror.com/api/datasets/DataScience-UIBK/RETECO-SemEval2027/tree/main"
RESOLVE = "https://hf-mirror.com/datasets/DataScience-UIBK/RETECO-SemEval2027/resolve/main"
ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "track1_tempo")


def list_dir(rel):
    r = requests.get(f"{BASE}/{rel}", timeout=60)
    r.raise_for_status()
    return r.json()


def download(rel, size):
    dst = os.path.join(DATA, *rel.split("/")[-2:]) if rel.startswith("track1_tempo/") \
        else os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with requests.get(f"{RESOLVE}/{rel}", stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    got = os.path.getsize(tmp)
    if size and got != size:
        os.replace(tmp, dst + ".badsize")
        raise RuntimeError(f"{rel}: size {got} != {size}")
    os.replace(tmp, dst)
    return dst


def verify(path):
    if path.endswith(".jsonl"):
        n = 0
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    json.loads(ln)
                    n += 1
        return f"{n} jsonl rows"
    if path.endswith(".txt"):
        n = 0
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    parts = ln.split()
                    if len(parts) != 4:
                        raise RuntimeError(f"{path}: bad qrels line {ln[:60]!r}")
                    n += 1
        return f"{n} qrels rows"
    return "binary"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doms = args.domains or sorted(d for d in os.listdir(DATA)
                                  if os.path.isdir(os.path.join(DATA, d)))
    report = []
    for d in doms:
        try:
            entries = list_dir(f"track1_tempo/{d}")
        except Exception as exc:  # noqa: BLE001
            print(f"{d}: LIST FAILED {exc}", flush=True)
            continue
        missing = []
        for e in entries:
            if e["type"] != "file":
                continue
            name = e["path"].split("/")[-1]
            local = os.path.join(DATA, d, name)
            have = os.path.getsize(local) if os.path.exists(local) else -1
            if have != e["size"]:
                missing.append((name, e["size"], have))
        if not missing:
            print(f"{d}: complete ({len(entries)} files)", flush=True)
            continue
        print(f"{d}: missing/differing -> " +
              ", ".join(f"{n}({s}B)" for n, s, _ in missing), flush=True)
        for name, size, have in missing:
            rel = f"track1_tempo/{d}/{name}"
            if args.dry_run:
                continue
            path = download(rel, size)
            info = verify(path)
            report.append({"domain": d, "file": name, "bytes": size, "check": info})
            print(f"  ok {d}/{name} {size}B {info}", flush=True)
    if report and not args.dry_run:
        out = os.path.join(ROOT, "results", "fetch_missing_train.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {out} ({len(report)} files)")


if __name__ == "__main__":
    main()
