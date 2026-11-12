#!/usr/bin/env python3
"""Fuse a dense run with the *average* of several cross-encoder runs.

Motivation: we trained the same recipe under three seeds (17, 23, 29) and got
0.3581 / 0.3575 / 0.3528 -- a spread of 0.003, i.e. the checkpoints disagree.
Averaging their z-normalised scores before fusing is the cheapest way to ask
whether the disagreement is noise that cancels.

    s(d) = wa * z(dense(d)) + wb * mean_k z(ce_k(d))

All z-scores are computed within a query over the dense candidate list, exactly
like scripts/fuse_scores.py, so the candidate set and the 1:1 interpretation are
unchanged. No qrels are read; nothing is tuned.

Usage:
  python scripts/fuse_ensemble.py \
      --dense "runs/dense_mxbai_{d}_dev.trec" \
      --ce "runs/rerank_sft_e2bc_{d}_dev.trec" \
      --ce "runs/rerank_sft_s23_{d}_dev.trec" \
      --ce "runs/rerank_sft_s29_{d}_dev.trec" \
      --out "runs/fused_ens3_{d}_dev.trec" \
      --json "results/fuse_ens3_{d}_dev.json" \
      --tag ens3 --wa 1 --wb 1
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import read_trec_run, write_trec_run  # noqa: E402

ROOT = os.environ.get("RETECO_ROOT", r"D:\RETECO")
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]


def z(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, help="template with {d}")
    ap.add_argument("--ce", action="append", required=True, help="template with {d}, repeatable")
    ap.add_argument("--out", required=True, help="template with {d}")
    ap.add_argument("--json", required=True, help="template with {d}")
    ap.add_argument("--tag", default="ens")
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    args = ap.parse_args()

    total = {"queries": 0, "missing_ce_docs": 0, "domains": 0}
    for d in [x for x in args.domains.split(",") if x]:
        dense_path = os.path.join(ROOT, args.dense.format(d=d))
        if not os.path.exists(dense_path):
            print("skip %s (no dense run)" % d, file=sys.stderr)
            continue
        A = read_trec_run(dense_path)
        ces = []
        for tpl in args.ce:
            p = os.path.join(ROOT, tpl.format(d=d))
            if not os.path.exists(p):
                print("missing CE run %s" % p, file=sys.stderr)
                continue
            ces.append(read_trec_run(p))
        if not ces:
            continue

        fused = {}
        for qid, rows in A.items():
            rows = sorted(rows, key=lambda r: r[0])
            amap = {doc: sc for _r, sc, doc in rows}
            # candidate set = what the frozen pipeline scored: the CE run's rows
            first = sorted(ces[0].get(qid, []), key=lambda r: r[0])
            cand = [doc for _r, _sc, doc in first if doc in amap]
            if not cand:
                continue
            a = z([amap[doc] for doc in cand])
            acc = np.zeros(len(cand))
            used = 0
            for C in ces:
                crows = C.get(qid, [])
                cmap = {doc: sc for _r, sc, doc in crows}
                vals = [cmap.get(doc) for doc in cand]
                hit = sum(1 for v in vals if v is not None)
                if hit == 0:
                    continue
                present = [v for v in vals if v is not None]
                lo = min(present)
                filled = np.array([v if v is not None else lo for v in vals])
                total["missing_ce_docs"] += len(cand) - hit
                acc += z(filled)
                used += 1
            if used == 0:
                continue
            ce = acc / used
            if ce.std() > 0:
                ce = z(ce)
            sc = args.wa * a + args.wb * ce
            order = np.argsort(-sc, kind="stable")
            fused[qid] = [(cand[i], float(sc[i])) for i in order]
            total["queries"] += 1

        out = os.path.join(ROOT, args.out.format(d=d))
        write_trec_run(out, fused, tag=args.tag)
        js = {"tag": args.tag, "domain": d, "queries": len(fused),
              "ce_runs": len(ces), "wa": args.wa, "wb": args.wb,
              "out": out, "ce_templates": args.ce, "dense": args.dense}
        with io.open(os.path.join(ROOT, args.json.format(d=d)), "w", encoding="utf-8") as f:
            json.dump(js, f, indent=1)
        total["domains"] += 1
        print("%-10s queries=%3d ce_runs=%d" % (d, len(fused), len(ces)))
    print(json.dumps(total))


if __name__ == "__main__":
    sys.exit(main())
