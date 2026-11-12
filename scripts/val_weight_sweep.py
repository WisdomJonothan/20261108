#!/usr/bin/env python3
"""Choose the fusion weight on the 45 held-out training queries, not on dev.

For each model the same z-fusion is swept on val, the argmax is recorded, and
the dev result at that weight is reported from results/CE_FUSION_PROBE.json so
the choice is made on one split and evaluated on another.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ir_metrics import ndcg_at_k  # noqa: E402
from lib_reteco import load_qrels, read_trec_run  # noqa: E402

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


def z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/RETECO")
    ap.add_argument("--models", nargs="+", default=["lx_possum", "cx_qv"])
    ap.add_argument("--qids", default="results/val_qids_e2.json")
    ap.add_argument("--out", default="results/VAL_WEIGHT_SWEEP.json")
    args = ap.parse_args()

    with open(os.path.join(args.root, args.qids), encoding="utf-8") as f:
        val = json.load(f)["eval"]

    report = {}
    for m in args.models:
        per_w = {w: {} for w in WEIGHTS}
        n_q = 0
        for d in DOMAINS:
            qrels = load_qrels(os.path.join(args.root, "data", "track1_tempo", d), "train")
            want = set(val.get(d, []))
            dense = read_trec_run(os.path.join(args.root, "runs", f"dense_mxbai_{d}_valpool.trec"))
            ce = read_trec_run(os.path.join(args.root, "runs", f"rerank_{m}_{d}_val.trec"))
            for q, rows in ce.items():
                sd = {doc: s for _, s, doc in dense.get(q, [])}
                keep = [(doc, s) for _, s, doc in rows if doc in sd]
                if not keep or q not in want:
                    continue
                docs = [x for x, _ in keep]
                sda = np.array([sd[x] for x, _ in keep])
                sca = np.array([s for _, s in keep])
                gold = set(qrels.get(q, []))
                for w in WEIGHTS:
                    s = z(sda) + w * z(sca)
                    order = [docs[i] for i in np.argsort(-s, kind="stable")]
                    per_w[w][(d, q)] = float(ndcg_at_k(order, gold, 10))
                n_q += 1
        macro = {}
        for w in WEIGHTS:
            by = {}
            for (d, q), v in per_w[w].items():
                by.setdefault(d, []).append(v)
            macro[w] = float(np.mean([np.mean(v) for v in by.values()])) if by else 0.0
        best = max(macro, key=macro.get)
        report[m] = {"queries": n_q, "macro": {str(w): macro[w] for w in WEIGHTS},
                     "best_w": best, "best_val": macro[best]}

    with open(os.path.join(args.root, args.out), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for m, r in report.items():
        print(m, {k: round(v, 4) for k, v in r["macro"].items()})
        print(f"  best w on val = {r['best_w']:g}  val {r['best_val']:.4f}  (n={r['queries']})")


if __name__ == "__main__":
    main()
