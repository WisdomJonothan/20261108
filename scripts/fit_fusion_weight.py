#!/usr/bin/env python3
"""Fit the dense:CE fusion configuration on the TRAIN split.

The dev split has already been the selection set for V1's design choices; every
extra choice made on dev shrinks the honest estimate of the next system. This
script therefore does the selection the way the shared task asks for it: score
every fusion candidate on `train`, pick the winner there, and let the caller
apply it to `dev` exactly once.

Candidates are (fusion form, weight) pairs:
  * `rrf`       — reciprocal rank fusion, `scripts/fuse_rrf.py` semantics
  * `z`         — within-query z-normalised score sum, `scripts/fuse_scores.py`
Weights are `w_dense:w_ce` in `1:1, 2:1, 4:1, 1:2, 1:4, 1:8`.

Reports, per candidate: topic-macro nDCG@10 over all train queries, per-domain
rows, and the paired bootstrap against the best fixed RRF baseline — all on
**train only**. No dev run is read.

Usage:
  python fit_fusion_weight.py --split train --out results/fusion_fit_train.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import read_trec_run, load_qrels  # noqa: E402
from ir_metrics import ndcg_at_k  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = ["4:1", "2:1", "1:1", "1:2", "1:4", "1:8"]


def zscore(v):
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def fused_order(dense, ce, form, wa, wb, c=60.0):
    """Return the fused ranking of the dense candidate set."""
    sa = {d: s for d, s in dense}
    sb = {d: s for d, s in ce}
    common = [d for d, _ in dense if d in sb]
    if form == "z":
        da = dict(zip(common, zscore([sa[d] for d in common])))
        db = dict(zip(common, zscore([sb[d] for d in common])))
        book = {d: wa * da[d] + wb * db[d] for d in common}
        order = sorted(book, key=lambda d: -book[d])
        return order + [d for d, _ in dense if d not in book]
    # rrf
    ra = {d: i + 1 for i, (d, _) in enumerate(dense)}
    rb = {d: i + 1 for i, (d, _) in enumerate(ce)}
    book = {}
    for d in ra:
        s = wa / (c + ra[d])
        if d in rb:
            s += wb / (c + rb[d])
        book[d] = s
    for d in rb:
        if d not in book:
            book[d] = wb / (c + rb[d])
    return sorted(book, key=lambda d: -book[d])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--dense-pattern", default="runs/dense_mxbai_{domain}_%s.trec")
    ap.add_argument("--ce-pattern", default="runs/rerank_mxbai_{domain}_%s.trec")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--forms", nargs="+", default=["rrf", "z"])
    ap.add_argument("--select-metric", choices=["domain_macro", "topic_macro"],
                    default="domain_macro",
                    help="official RETECO 1a aggregation is domain-macro "
                         "(mean within domain, then mean over the 13 domains)")
    ap.add_argument("--out", default="results/fusion_fit_train.json")
    args = ap.parse_args()

    doms = args.domains or sorted(d for d in os.listdir(os.path.join(ROOT, "data", "track1_tempo"))
                                  if os.path.isdir(os.path.join(ROOT, "data", "track1_tempo", d)))
    dense_pat = (args.dense_pattern % args.split) if "%s" in args.dense_pattern else args.dense_pattern
    ce_pat = (args.ce_pattern % args.split) if "%s" in args.ce_pattern else args.ce_pattern

    # per-query nDCG@10 for every (form, weight)
    scores = {(f, w): {} for f in args.forms for w in WEIGHTS}
    topics, skipped = 0, []
    for d in doms:
        dp = os.path.join(ROOT, dense_pat.format(domain=d))
        cp = os.path.join(ROOT, ce_pat.format(domain=d))
        if not (os.path.exists(dp) and os.path.exists(cp)):
            skipped.append(d)
            continue
        dense = read_trec_run(dp)
        ce = read_trec_run(cp)
        qrels = load_qrels(os.path.join(ROOT, "data", "track1_tempo", d), args.split)
        for qid, rows in dense.items():
            if qid not in qrels:
                continue
            gold = qrels[qid]
            cand = [(t[2], t[1]) for t in rows][:args.k]
            cerows = [(t[2], t[1]) for t in ce.get(qid, [])]
            topics += 1
            for f in args.forms:
                for w in WEIGHTS:
                    wa, wb = (float(x) for x in w.split(":"))
                    order = fused_order(cand, cerows, f, wa, wb)
                    scores[(f, w)][(d, qid)] = ndcg_at_k(order, gold, 10)

    rows = []
    for (f, w), per in scores.items():
        if not per:
            continue
        vals = np.array(list(per.values()))
        per_dom = {}
        for (d, _), v in per.items():
            per_dom.setdefault(d, []).append(v)
        rows.append({"form": f, "weights": w,
                     "topic_macro": float(vals.mean()),
                     "domain_macro": float(np.mean([np.mean(v) for v in per_dom.values()])),
                     "n": len(vals),
                     "per_domain": {k: float(np.mean(v)) for k, v in per_dom.items()}})
    rows.sort(key=lambda r: -r[args.select_metric])

    print(f"train: {topics} topics, {len(doms)-len(skipped)} domains"
          + (f", skipped {skipped}" if skipped else ""))
    print(f"{'form':5s} {'w_d:w_c':8s} {'topic-macro':>11s} {'domain-macro':>12s}")
    for r in rows:
        print(f"{r['form']:5s} {r['weights']:8s} {r['topic_macro']:>11.4f} {r['domain_macro']:>12.4f}")
    best = rows[0]
    print(f"\nBEST on train: form={best['form']} weights={best['weights']} "
          f"topic-macro={best['topic_macro']:.4f}")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "topics": topics, "skipped_domains": skipped,
                   "select_metric": args.select_metric,
                   "candidates": rows, "best": best}, f, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
