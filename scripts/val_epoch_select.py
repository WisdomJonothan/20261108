#!/usr/bin/env python3
"""Compare the CE-only and fused validation score of every saved epoch.

The pipeline selects checkpoints on the cross-encoder's own validation nDCG,
but what gets submitted is the dense+CE fusion.  This script scores each saved
epoch on the 45 held-out training queries twice - once as the cross-encoder
would see it (CE ranking alone) and once after z-fusion with the frozen dense
run at w = 1 - and reports which epoch each objective would pick.
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


def domain_macro(rows):
    by = {}
    for d, v in rows:
        by.setdefault(d, []).append(v)
    if not by:
        return 0.0, 0
    return float(np.mean([np.mean(v) for v in by.values()])), sum(len(v) for v in by.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/RETECO")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--qids", default="results/val_qids_e2.json")
    ap.add_argument("--out", default="results/VAL_EPOCH_SELECT.json")
    args = ap.parse_args()

    with open(os.path.join(args.root, args.qids), encoding="utf-8") as f:
        val = json.load(f)["eval"]

    report = {}
    for tag in args.tags:
        ce_rows, fu_rows, cd_rows = [], [], []
        for d in DOMAINS:
            qrels = load_qrels(os.path.join(args.root, "data", "track1_tempo", d), "train")
            want = set(val.get(d, []))
            dense = read_trec_run(os.path.join(args.root, "runs", f"dense_mxbai_{d}_valpool.trec"))
            ce = read_trec_run(os.path.join(args.root, "runs", f"rerank_{tag}_{d}_val.trec"))
            for q, rows in ce.items():
                if q not in want:
                    continue
                gold = set(qrels.get(q, []))
                ce_order = [doc for _, _, doc in rows]
                ce_rows.append((d, float(ndcg_at_k(ce_order, gold, 10))))
                sd = {doc: s for _, s, doc in dense.get(q, [])}
                keep = [(doc, s) for _, s, doc in rows if doc in sd]
                if not keep:
                    fu_rows.append((d, 0.0))
                    continue
                docs = [x for x, _ in keep]
                fused = z([sd[x] for x in docs]) + args.w * z([s for _, s in keep])
                order = [docs[i] for i in np.argsort(-fused, kind="stable")]
                fu_rows.append((d, float(ndcg_at_k(order, gold, 10))))
                cd_rows.append((d, float(len(set(ce_order[:10]) & set(order[:10])) / 10.0)))
        ce_m, n = domain_macro(ce_rows)
        fu_m, _ = domain_macro(fu_rows)
        cd_m, _ = domain_macro(cd_rows)
        report[tag] = {"queries": n, "ce_only_macro@10": ce_m,
                       "fused_macro@10": fu_m, "top10_overlap_with_fused": cd_m}

    with open(os.path.join(args.root, args.out), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"{'epoch':>10}  {'CE-only':>8}  {'fused':>8}  {'top10∩':>7}  n")
    for tag, r in report.items():
        print(f"{tag:>10}  {r['ce_only_macro@10']:8.4f}  {r['fused_macro@10']:8.4f}  "
              f"{r['top10_overlap_with_fused']:7.3f}  {r['queries']}")
    best_ce = max(report, key=lambda t: report[t]["ce_only_macro@10"])
    best_fu = max(report, key=lambda t: report[t]["fused_macro@10"])
    print(f"\nCE-only pick: {best_ce}   fused pick: {best_fu}"
          + ("" if best_ce == best_fu else "   <-- the two objectives disagree"))


if __name__ == "__main__":
    main()
