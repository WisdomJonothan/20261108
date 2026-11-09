#!/usr/bin/env python3
"""Post-hoc weight sensitivity of the SFT-vs-zero-shot fusion comparison.

The published fusion weight (z-sum 1:1) was fitted ONCE on the 1,211 train
queries using the *zero-shot* CE scores, then frozen; the SFT arm inherited it.
This sweep answers the reviewer's actual question -- "is the +0.0087 SFT gain an
artefact of that weight?" -- instead of re-fitting an optimiser nobody asked
for: it re-fuses the stored per-query dense/CE scores at a range of weights for
BOTH arms on dev and reports the delta, with a domain-clustered bootstrap CI.

It is diagnostic only. Nothing is selected here.

Usage:
  python scripts/weight_sweep.py --out results/weight_sweep_dev.json \
      --md results/WEIGHT_SWEEP.md
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_qrels, read_trec_run  # noqa: E402

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]
WEIGHTS = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.0]


def ndcg_at_10(order, gold):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(order[:10]) if d in gold)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg else 0.0


def zscore(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def fuse_order(dense, ce, w_ce):
    docs = [d for d in dense if d in ce]
    if not docs:
        return []
    s = zscore([dense[d] for d in docs]) + w_ce * zscore([ce[d] for d in docs])
    return [docs[i] for i in np.argsort(-s, kind="stable")]


def spearman(a, b):
    """Rank correlation over the shared candidate list."""
    docs = [d for d in a if d in b]
    ra = {d: i for i, d in enumerate(docs)}
    rb = {d: i for i, d in enumerate(sorted(docs, key=lambda d: -b.get(d, -1e9)))}
    n = len(docs)
    if n < 3:
        return float("nan")
    da = np.array([ra[d] for d in docs], dtype=float)
    db = np.array([rb[d] for d in docs], dtype=float)
    da -= da.mean()
    db -= db.mean()
    den = np.sqrt((da ** 2).sum() * (db ** 2).sum())
    return float((da * db).sum() / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dense", default="runs/dense_mxbai_{d}_{s}.trec")
    ap.add_argument("--zs", default="runs/rerank_mxbai_{d}_{s}.trec")
    ap.add_argument("--sft", default="runs/rerank_sftv1_{d}_{s}.trec")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/weight_sweep_dev.json")
    ap.add_argument("--md", default="results/WEIGHT_SWEEP.md")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    per_w = {w: {"zs": {}, "sft": {}} for w in WEIGHTS}
    per_q = {w: {"zs": {}, "sft": {}} for w in WEIGHTS}
    corr = {}
    for d in DOMAINS:
        ddir = os.path.join(root, "data", "track1_tempo", d)
        gold = {q: set(v) for q, v in load_qrels(ddir, args.split).items()}
        dense = read_trec_run(os.path.join(root, args.dense.format(d=d, s=args.split)))
        zs = read_trec_run(os.path.join(root, args.zs.format(d=d, s=args.split)))
        sft = read_trec_run(os.path.join(root, args.sft.format(d=d, s=args.split)))
        corr[d] = []
        for qid, gl in gold.items():
            if qid not in dense:
                continue
            dd = {doc: sc for _r, sc, doc in dense[qid]}
            zc = {doc: sc for _r, sc, doc in zs.get(qid, [])}
            sc_ = {doc: sc for _r, sc, doc in sft.get(qid, [])}
            if zc and sc_:
                corr[d].append(spearman(dd, sc_))
            for w in WEIGHTS:
                per_w[w]["zs"].setdefault(d, []).append(ndcg_at_10(fuse_order(dd, zc, w), gl))
                per_w[w]["sft"].setdefault(d, []).append(ndcg_at_10(fuse_order(dd, sc_, w), gl))
                per_q[w]["zs"].setdefault(d, []).append(ndcg_at_10(fuse_order(dd, zc, w), gl))
                per_q[w]["sft"].setdefault(d, []).append(ndcg_at_10(fuse_order(dd, sc_, w), gl))

    def macro(w, arm):
        return float(np.mean([np.mean(per_w[w][arm][d]) for d in DOMAINS]))

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(DOMAINS))
    out = {"weights": WEIGHTS, "split": args.split, "bootstrap": args.boot, "rows": []}
    for w in WEIGHTS:
        zs_m, sft_m = macro(w, "zs"), macro(w, "sft")
        # domain-clustered bootstrap of the delta
        deltas = []
        for _ in range(args.boot):
            pick = rng.choice(idx, size=len(idx), replace=True)
            dd = [np.mean(per_w[w]["sft"][DOMAINS[i]]) - np.mean(per_w[w]["zs"][DOMAINS[i]])
                  for i in pick]
            deltas.append(float(np.mean(dd)))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        wins = sum(1 for d in DOMAINS
                   if np.mean(per_w[w]["sft"][d]) > np.mean(per_w[w]["zs"][d]))
        out["rows"].append({"w_ce": w, "zs": zs_m, "sft": sft_m, "delta": sft_m - zs_m,
                            "ci_lo": float(lo), "ci_hi": float(hi), "domains_up": wins})
    out["spearman_dense_vs_sft_ce"] = float(np.nanmean([np.nanmean(v) for v in corr.values()]))

    w1 = next(r for r in out["rows"] if r["w_ce"] == 1.0)
    out["per_domain_at_1_1"] = [
        {"domain": d,
         "zs": float(np.mean(per_w[1.0]["zs"][d])),
         "sft": float(np.mean(per_w[1.0]["sft"][d])),
         "delta": float(np.mean(per_w[1.0]["sft"][d]) - np.mean(per_w[1.0]["zs"][d]))}
        for d in DOMAINS]
    out["summary"] = w1
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    lines = ["# Weight sensitivity of the SFT gain (dev, domain-macro nDCG@10)", "",
             "Fusion: `s = z(dense) + w * z(CE)` per query, same candidate set. The",
             "published weight is w=1; it was fitted on train with the ZERO-SHOT CE only.",
             "Diagnostic sweep, nothing selected.", "",
             "| w (CE) | zero-shot | SFT | delta | 95% domain-clustered CI | domains up |",
             "|---:|---:|---:|---:|---|---:|"]
    for r in out["rows"]:
        lines.append(f"| {r['w_ce']:g} | {r['zs']:.4f} | {r['sft']:.4f} | "
                     f"{r['delta']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                     f"{r['domains_up']}/13 |")
    lines += ["", f"Mean Spearman (dense vs SFT CE) over the top-100: "
                  f"{out['spearman_dense_vs_sft_ce']:.3f}.", "",
              "## Per domain at the published weight (w=1)", "",
              "| domain | zero-shot | SFT | delta |", "|---|---:|---:|---:|"]
    for r in out["per_domain_at_1_1"]:
        lines.append(f"| {r['domain']} | {r['zs']:.4f} | {r['sft']:.4f} | {r['delta']:+.4f} |")
    with open(args.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(json.dumps({"summary": out["summary"],
                      "spearman": out["spearman_dense_vs_sft_ce"]}, indent=1))


if __name__ == "__main__":
    main()
