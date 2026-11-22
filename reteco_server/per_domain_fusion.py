"""Per-domain fusion weights, fitted on the leak-free train folds and applied to dev.

Motivation: the frozen system fuses dense and cross-encoder scores 1:1 in every domain,
but the internal evidence says the cross-encoder helps in some domains and hurts in
others. The weight is a one-parameter decision, so fitting it per domain on the 969 train
queries is cheap and auditable.

Leak-freeness: the cross-encoder scores used for fitting are the out-of-fold predictions
(`runs/<fold>_ce_<domain>_train_ce.trec` from the three gate folds), so no train query is
scored by a model that saw it. Dev is only read once, with the weights fixed from train.

Usage (server):
  .venv/bin/python scripts/per_domain_fusion.py --root .
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DOMAINS = [
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
    "law", "monero", "politics", "quant", "travel", "workplace",
]
GRID = [round(0.0 + 0.25 * i, 2) for i in range(9)]  # 0.00 .. 2.00


def read_run(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, docid = parts[0], parts[2]
            score = float(parts[4]) if len(parts) >= 6 else float(parts[3])
            out.setdefault(qid, {})[docid] = score
    return out


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            docid = parts[1] if len(parts) == 3 else parts[2]
            grade = int(float(parts[-1]))
            if grade > 0:
                out.setdefault(qid, {})[docid] = grade
    return out


def ndcg10(ranked: list[str], gold: dict[str, int]) -> float:
    dcg = sum(gold.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(gold.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def fuse_z(dense: dict[str, float], ce: dict[str, float], w: float) -> list[str]:
    docs = set(dense) & set(ce)
    if not docs:
        return sorted(dense, key=lambda d: -dense[d])[:10]
    dvals = np.array([dense[d] for d in docs])
    cvals = np.array([ce[d] for d in docs])
    dz = (dvals - dvals.mean()) / (dvals.std() + 1e-9)
    cz = (cvals - cvals.mean()) / (cvals.std() + 1e-9)
    scored = sorted(zip(docs, dz + w * cz), key=lambda kv: -kv[1])
    return [d for d, _ in scored][:10]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--folds", default="f0,f1,f2")
    ap.add_argument("--out", default="results/analysis/per_domain_fusion.json")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    folds = args.folds.split(",")

    report: dict[str, dict] = {}
    dev_macro = {f"w={w}": [] for w in GRID}
    dev_macro["frozen_w1"] = []
    fitted_macro = []

    for dom in DOMAINS:
        dense_tr = read_run(root / "runs" / f"dense_mxbai_{dom}_train.trec")
        oof: dict[str, dict[str, float]] = {}
        for f in folds:
            p = root / "runs" / f"{f}_ce_{dom}_train_ce.trec"
            if p.exists():
                for qid, docs in read_run(p).items():
                    oof.setdefault(qid, {}).update(docs)
        gold_tr = read_qrels(root / "data" / "track1_tempo" / dom / "qrels_train.txt")

        qids = [q for q in gold_tr if q in dense_tr and q in oof]
        best_w, best_score = None, -1.0
        scores_by_w = {}
        for w in GRID:
            vals = [ndcg10(fuse_z(dense_tr[q], oof[q], w), gold_tr[q]) for q in qids]
            m = float(np.mean(vals)) if vals else float("nan")
            scores_by_w[w] = m
            if vals and m > best_score:
                best_w, best_score = w, m
        if not qids:
            best_w = 1.0

        # dev, with the weight fixed from train
        dense_dv = read_run(root / "runs" / f"dense_mxbai_{dom}_dev.trec")
        ce_dv = read_run(root / "runs" / f"fusion100_full_ce_s17_20261115_{dom}_dev_ce.trec")
        gold_dv = read_qrels(root / "data" / "track1_tempo" / dom / "qrels_dev.txt")
        dev_qids = [q for q in gold_dv if q in dense_dv and q in ce_dv]

        fitted_vals = [ndcg10(fuse_z(dense_dv[q], ce_dv[q], best_w), gold_dv[q]) for q in dev_qids]
        frozen_vals = [ndcg10(fuse_z(dense_dv[q], ce_dv[q], 1.0), gold_dv[q]) for q in dev_qids]
        for w in GRID:
            dev_macro[f"w={w}"].append(float(np.mean([ndcg10(fuse_z(dense_dv[q], ce_dv[q], w), gold_dv[q])
                                                     for q in dev_qids])))
        dev_macro["frozen_w1"].append(float(np.mean(frozen_vals)))
        fitted_macro.append(float(np.mean(fitted_vals)))

        report[dom] = {"n_train": len(qids), "n_dev": len(dev_qids),
                       "train_curve": scores_by_w, "best_w": best_w,
                       "train_at_best": best_score,
                       "dev_at_best": float(np.mean(fitted_vals)),
                       "dev_at_w1": float(np.mean(frozen_vals)),
                       "delta_dev": float(np.mean(fitted_vals) - np.mean(frozen_vals))}
        print(dom, "best_w", best_w, "train", round(best_score, 4),
              "dev", round(report[dom]["dev_at_best"], 4),
              "vs w1", round(report[dom]["delta_dev"], 4), flush=True)

    summary = {
        "fitted_per_domain": float(np.mean(fitted_macro)),
        "frozen_w1": float(np.mean(dev_macro["frozen_w1"])),
        "grid_dev_macro": {k: float(np.mean(v)) for k, v in dev_macro.items() if k != "frozen_w1"},
        "best_w_by_domain": {d: report[d]["best_w"] for d in DOMAINS},
    }
    payload = {"grid": GRID, "summary": summary, "per_domain": report}
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("SUMMARY", json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in summary.items() if k != "grid_dev_macro"}))
    print("grid", {k: round(v, 4) for k, v in summary["grid_dev_macro"].items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
