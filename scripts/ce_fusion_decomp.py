#!/usr/bin/env python3
"""Decompose the CE-versus-fusion gap into the three channels of S5.8.

For every dev query, score the cross-encoder ordering alone and the z-fused
ordering, for the reference arm (`lx_possum`) and the probe arm (`cx_qv`), then
group queries by how much the CE top ten changed:

  improved  - CE nDCG@10 up by more than 0.01
  same      - CE top ten unchanged (|delta| <= 0.01), fusion still moves
  hurt      - CE nDCG@10 down by more than 0.01

and fit the slope of the fused delta on the CE delta.  Writes
results/CE_FUSION_DECOMP.json.
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


def z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/RETECO")
    ap.add_argument("--ref", default="lx_possum")
    ap.add_argument("--probe", default="cx_qv")
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--out", default="results/CE_FUSION_DECOMP.json")
    args = ap.parse_args()

    rows = []
    for d in DOMAINS:
        qrels = load_qrels(os.path.join(args.root, "data", "track1_tempo", d), "dev")
        dense = read_trec_run(os.path.join(args.root, "runs", f"dense_mxbai_{d}_dev.trec"))
        ref = read_trec_run(os.path.join(args.root, "runs", f"rerank_{args.ref}_{d}_dev.trec"))
        prb = read_trec_run(os.path.join(args.root, "runs", f"rerank_{args.probe}_{d}_dev.trec"))
        for q, rows_a in ref.items():
            rows_b = prb.get(q)
            if not rows_b:
                continue
            gold = set(qrels.get(q, []))
            if not gold:
                continue
            sd = {doc: s for _, s, doc in dense.get(q, [])}
            out = {}
            for name, rr in (("ref", rows_a), ("probe", rows_b)):
                keep = [(doc, s) for _, s, doc in rr if doc in sd]
                docs = [x for x, _ in keep]
                ce = [s for _, s in keep]
                ce_order = [docs[i] for i in np.argsort(-np.array(ce), kind="stable")]
                fused = z([sd[x] for x in docs]) + args.w * z(ce)
                fu_order = [docs[i] for i in np.argsort(-fused, kind="stable")]
                out[name] = {"ce": float(ndcg_at_k(ce_order, gold, 10)),
                             "fused": float(ndcg_at_k(fu_order, gold, 10)),
                             "ce_top10": set(ce_order[:10])}
            de = out["probe"]["ce"] - out["ref"]["ce"]
            df = out["probe"]["fused"] - out["ref"]["fused"]
            same = len(out["probe"]["ce_top10"] & out["ref"]["ce_top10"]) / 10.0
            rows.append({"domain": d, "qid": q, "dce": de, "dfused": df,
                         "ce_top10_overlap": same})

    groups = {"improved": [], "same": [], "hurt": []}
    for r in rows:
        if r["dce"] > 0.01:
            groups["improved"].append(r)
        elif r["dce"] < -0.01:
            groups["hurt"].append(r)
        else:
            groups["same"].append(r)

    report = {"ref": args.ref, "probe": args.probe, "w": args.w, "queries": len(rows)}
    for name, g in groups.items():
        report[name] = {
            "n": len(g),
            "mean_dce": float(np.mean([r["dce"] for r in g])) if g else 0.0,
            "mean_dfused": float(np.mean([r["dfused"] for r in g])) if g else 0.0,
            "sum_dfused": float(np.sum([r["dfused"] for r in g])) if g else 0.0,
            "mean_ce_top10_overlap": float(np.mean([r["ce_top10_overlap"] for r in g])) if g else 0.0,
        }
    x = np.array([r["dce"] for r in rows])
    y = np.array([r["dfused"] for r in rows])
    slope = float(np.polyfit(x, y, 1)[0])
    report["slope_dfused_on_dce"] = slope
    report["mean_dfused_all"] = float(y.mean())
    with open(os.path.join(args.root, args.out), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    for name in ("improved", "same", "hurt"):
        g = report[name]
        print(f"{name:>9} n={g['n']:3d}  dCE {g['mean_dce']:+.4f}  dFused {g['mean_dfused']:+.4f}"
              f"  top10 overlap {g['mean_ce_top10_overlap']:.2f}")
    print(f"slope = {slope:.3f}  mean dfused = {report['mean_dfused_all']:+.4f}")


if __name__ == "__main__":
    main()
