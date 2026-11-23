#!/usr/bin/env python3
"""Four-cell requirement experiment: out-of-fold comparison.

Cell A is the frozen six-pair configuration replayed from the judgement cache;
cells sel / req / both / mis are the re-judged variants.  Everything is reported
per fold and pooled, against (i) the frozen ranking and (ii) cell A, with paired
bootstrap intervals (10k draws, queries resampled within domain).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CELLS = ["base", "sel", "req", "both", "mis"]


def six_rule(actions):
    pool = sorted(actions, key=lambda a: (a["slot"], -a["prior"]))[:6]
    elig = [a for a in pool if a.get("agree") and a["mp"] > 0.5]
    return max(elig, key=lambda a: a["mp"]) if elig else None


def domain_macro(rows, key):
    per = defaultdict(list)
    for r in rows:
        per[r["domain"]].append(r[key])
    doms = sorted(per)
    return float(np.mean([np.mean(per[d]) for d in doms])), doms


def paired(rows, a_key, b_key, draws=10000, seed=20261120):
    by_dom = defaultdict(list)
    for r in rows:
        by_dom[r["domain"]].append(r[b_key] - r[a_key])
    doms = sorted(by_dom)
    point = float(np.mean([np.mean(by_dom[d]) for d in doms]))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        acc = [np.array(by_dom[d])[rng.integers(0, len(by_dom[d]), len(by_dom[d]))].mean()
               for d in doms]
        stats.append(float(np.mean(acc)))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi), float(np.mean(np.array(stats) <= 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--folds", default="f0,f1,f2")
    ap.add_argument("--out", default="results/replacement_v2/req/summary.json")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    R = root / "results" / "replacement_v2"

    pooled = {c: [] for c in CELLS}
    per_fold = {}
    for fold in a.folds.split(","):
        cache = json.load(open(R / f"expanded3_fixed_train_{fold}.json", encoding="utf-8"))
        cached = {r["qid"]: r for r in cache["budgets"]["18"]["per_query"]}
        cells = {}
        for c in CELLS:
            if c == "base":
                continue
            p = R / "req" / f"{c}_{fold}.json"
            if p.exists():
                cells[c] = {r["qid"]: r for r in json.load(open(p, encoding="utf-8"))["per_query"]}
        if not cells:
            continue
        qids = set.intersection(*[set(v) for v in cells.values()]) & set(cached)
        rows = []
        for q in sorted(qids):
            base_row = cached[q]
            best = six_rule(base_row["actions"])
            rows.append({"domain": base_row["domain"], "qid": q,
                         "base": base_row["baseline"],
                         "base_six": base_row["baseline"] + (best["delta"] if best else 0.0)})
        for c, data in cells.items():
            for r in rows:
                r[c] = data[r["qid"]]["cell"]
        pooled["base"] += rows
        for c in cells:
            pooled[c] += rows
        m = {c: domain_macro(rows, "base" if c == "base" else c)[0] for c in ["base", "base_six", *cells]}
        per_fold[fold] = {"n": len(rows), "macro": m}
        print(fold, json.dumps({k: round(v, 6) for k, v in m.items()}))

    print("\n=== pooled ===")
    out = {"folds": per_fold, "cells": {}}
    base_macro, _ = domain_macro(pooled["base"], "base")
    six_macro, _ = domain_macro(pooled["base"], "base_six")
    print(f"queries {len(pooled['base'])}  frozen {base_macro:.6f}  six-pair cellA {six_macro:.6f} "
          f"({six_macro - base_macro:+.6f})")
    out["n_queries"] = len(pooled["base"])
    out["frozen"] = base_macro
    out["cellA"] = six_macro
    for c in CELLS:
        if c == "base" or not pooled[c]:
            continue
        rows = pooled[c]
        m, _ = domain_macro(rows, c)
        d_base = paired(rows, "base", c)
        d_six = paired(rows, "base_six", c)
        swaps = sum(1 for r in rows if abs(r[c] - r["base_six"]) > 1e-12)
        helped = sum(1 for r in rows if r[c] - r["base_six"] > 1e-9)
        hurt = sum(1 for r in rows if r[c] - r["base_six"] < -1e-9)
        recovered = sum(1 for r in rows if r["base_six"] - r["base"] <= 1e-12 and r[c] > r["base"])
        lost = sum(1 for r in rows if r["base_six"] > r["base"] and r[c] <= r["base"])
        harmful_added = sum(1 for r in rows if r["base_six"] >= r["base"] and r[c] < r["base"])
        harmful_removed = sum(1 for r in rows if r["base_six"] < r["base"] and r[c] >= r["base"])
        rec = {"macro": m, "delta_vs_frozen": m - base_macro, "delta_vs_cellA": m - six_macro,
               "paired_vs_frozen": d_base, "paired_vs_cellA": d_six,
               "queries_changed_vs_cellA": swaps, "helped": helped, "hurt": hurt,
               "beneficial_recovered": recovered, "beneficial_lost": lost,
               "harmful_added": harmful_added, "harmful_removed": harmful_removed}
        out["cells"][c] = rec
        print(f"{c:5s} {m:.6f}  vs frozen {m - base_macro:+.6f} "
              f"[{d_base[1]:+.6f},{d_base[2]:+.6f}]  vs cellA {m - six_macro:+.6f} "
              f"[{d_six[1]:+.6f},{d_six[2]:+.6f}]  changed {swaps} (+{helped}/-{hurt}) "
              f"recovered {recovered} lost {lost} harm+{harmful_added}/-{harmful_removed}")

    json.dump(out, open(root / a.out, "w"), indent=1)

    # matched-versus-mismatched requirement, the decisive contrast
    contrasts = {}
    base_rows = pooled["base"]
    if pooled["req"] and pooled["mis"]:
        have = [r for r in base_rows if "req" in r and "mis" in r]
        point, lo, hi, p = paired(have, "mis", "req")
        contrasts["req_minus_mis"] = {"delta": point, "ci95": [lo, hi], "p_le_0": p,
                                      "n": len(have)}
        print(f"req - mis {point:+.6f} [{lo:+.6f},{hi:+.6f}]  n={len(have)}  p={p:.4f}")
    if pooled["both"] and pooled["mis"]:
        have = [r for r in base_rows if "both" in r and "mis" in r]
        point, lo, hi, p = paired(have, "mis", "both")
        contrasts["both_minus_mis"] = {"delta": point, "ci95": [lo, hi], "p_le_0": p,
                                       "n": len(have)}
        print(f"both - mis {point:+.6f} [{lo:+.6f},{hi:+.6f}]  n={len(have)}  p={p:.4f}")
    if pooled["sel"]:
        have = [r for r in base_rows if "sel" in r and "base_six" in r]
        point, lo, hi, p = paired(have, "sel", "base_six")
        contrasts["sel_minus_cellA"] = {"delta": point, "ci95": [lo, hi], "p_le_0": p,
                                        "n": len(have)}
        print(f"sel - cellA {point:+.6f} [{lo:+.6f},{hi:+.6f}]  p={p:.4f}")
    out["contrasts"] = contrasts
    json.dump(out, open(root / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
