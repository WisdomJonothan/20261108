#!/usr/bin/env python3
"""Validate the six-pair head-first definition against the frozen dev number.

Cell A of the requirement experiment must be the same configuration that scores
0.373503 on dev.  Definition under test: of the 18 judged actions per query, sort
by slot ascending (ties by prior descending) and keep the first six; apply the
frozen acceptance rule (both orders agree, merged > 0.5, best merged, one swap).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def choose6(actions):
    pool = sorted(actions, key=lambda a: (a["slot"], -a["prior"]))[:6]
    elig = [a for a in pool if a.get("agree") and a["mp"] > 0.5]
    return max(elig, key=lambda a: a["mp"]) if elig else None


def macro(rows, key):
    per = defaultdict(list)
    for r in rows:
        per[r["domain"]].append(r[key])
    doms = sorted(per)
    return float(np.mean([np.mean(per[d]) for d in doms])), doms, per


def main() -> None:
    for name in ("expanded3_fixed_dev.json",):
        d = json.load(open(ROOT / "results" / "replacement_v2" / name))
        rows = d["budgets"]["18"]["per_query"]
        out = []
        for r in rows:
            a = choose6(r["actions"])
            out.append({"domain": r["domain"], "qid": r["qid"], "baseline": r["baseline"],
                        "policy": r["baseline"] + (a["delta"] if a else 0.0),
                        "swapped": bool(a), "delta": (a["delta"] if a else 0.0)})
        m, doms, per = macro(out, "policy")
        mb, _, _ = macro(out, "baseline")
        swaps = sum(1 for r in out if r["swapped"])
        help_ = sum(1 for r in out if r["swapped"] and r["delta"] > 1e-9)
        harm = sum(1 for r in out if r["swapped"] and r["delta"] < -1e-9)
        print(f"{name}: n={len(out)} baseline={mb:.6f} policy={m:.6f} "
              f"delta={m - mb:+.6f} swaps={swaps} helpful={help_} harmful={harm}")


if __name__ == "__main__":
    main()
