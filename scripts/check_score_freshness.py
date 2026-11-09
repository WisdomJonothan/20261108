#!/usr/bin/env python3
"""Detect score JSONs that are older than the run file they were computed from.

The Phase 11 incident (2026-11-09): the fused TREC files were regenerated at the
end of a rerank stage, the score JSONs were not, and a comparison against a
frozen number silently used a stale pair (0.3196 instead of 0.3583). This script
walks every `results/score_*.json`, reads the `run_file` recorded inside it, and
reports the pairs whose run file is newer than the score file.

Usage:
  python scripts/check_score_freshness.py             # report only
  python scripts/check_score_freshness.py --fix       # re-score stale pairs
  python scripts/check_score_freshness.py --json results/score_freshness.json

Exit code 0 when nothing is stale, 1 when stale files were found.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable


def stale_pairs(slack: float = 2.0) -> list[tuple[str, str]]:
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "score_*_dev.json"))):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        run = j.get("run_file")
        if not run:
            continue
        run = os.path.normpath(run)
        if not os.path.exists(run):
            out.append((p, "MISSING " + run))
            continue
        if os.path.getmtime(run) > os.path.getmtime(p) + slack:
            out.append((p, run))
    return out


def rescore(pair: tuple[str, str]) -> bool:
    path, run = pair
    j = json.load(open(path, encoding="utf-8"))
    cmd = [PY, os.path.join(ROOT, "scripts", "score_run.py"),
           "--run", run, "--data", j["data"], "--split", j.get("split", "dev"),
           "--json", path, "--tag", j.get("tag", "")]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONUTF8="1"))
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="re-score every stale file")
    ap.add_argument("--json", default=None)
    ap.add_argument("--slack", type=float, default=2.0, help="seconds of mtime slack")
    args = ap.parse_args()

    pairs = stale_pairs(args.slack)
    for p, run in pairs:
        print(f"STALE {os.path.relpath(p, ROOT)}  <-  {os.path.relpath(run, ROOT) if os.path.exists(run) else run}")
    if args.fix:
        ok = sum(rescore(p) for p in pairs)
        print(f"re-scored {ok}/{len(pairs)}")
        pairs = stale_pairs(args.slack)
        print(f"remaining stale: {len(pairs)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"stale": [{"score": p, "run": r} for p, r in pairs],
                       "checked": len(glob.glob(os.path.join(ROOT, 'results', 'score_*_dev.json')))},
                      f, indent=1)
    print("OK" if not pairs else f"{len(pairs)} stale")
    return 0 if not pairs else 1


if __name__ == "__main__":
    sys.exit(main())
