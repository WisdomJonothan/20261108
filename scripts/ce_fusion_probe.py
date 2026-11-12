#!/usr/bin/env python3
"""Diagnose why a better cross-encoder does not improve the fused system.

For every dev query we hold the dense top-100 (the frozen candidate list) and
two cross-encoder runs over exactly those 100 documents.  The frozen fusion is

    s(d) = z_dense(d) + w * z_ce(d)      (z within the query, over the 100
                                          common candidates)

so a change in the cross-encoder can reach the fused list in two ways: it can
*reorder* the candidates, or it can change the *shape* of the z profile (the
score gap it leaves between neighbouring ranks).  This script separates those
and sweeps the CE weight.  No training, no GPU.

Modes
  dense / ce      w = 0 and w = inf, on the same 100-document pool
  sweep           fused = z_d + w z_ce for a grid of w, per model
  transplant      ordering from one model, z profile from the other
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


def load_pool(root, dense_tag, ce_tag, domain):
    """{qid: (docs, dense_scores, ce_scores)} aligned on the CE's 100 docs."""
    dense = read_trec_run(os.path.join(root, "runs", f"dense_{dense_tag}_{domain}_dev.trec"))
    ce = read_trec_run(os.path.join(root, "runs", f"rerank_{ce_tag}_{domain}_dev.trec"))
    out = {}
    for qid, rows in ce.items():
        sd = {d: s for _, s, d in dense.get(qid, [])}
        keep = [(d, s) for _, s, d in rows if d in sd]
        if not keep:
            continue
        out[qid] = ([d for d, _ in keep],
                    np.array([sd[d] for d, _ in keep], dtype=float),
                    np.array([s for _, s in keep], dtype=float))
    return out


def ndcg_of(order, gold):
    return float(ndcg_at_k(order, set(gold), 10)) if gold else 0.0


def macro(per_domain):
    return float(np.mean([np.mean(list(per_domain[d].values())) for d in DOMAINS if per_domain[d]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/RETECO")
    ap.add_argument("--dense-tag", default="mxbai")
    ap.add_argument("--models", nargs="+", default=["lx_possum", "cx_qv"])
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0])
    ap.add_argument("--out", default="D:/RETECO/results/CE_FUSION_PROBE.json")
    ap.add_argument("--emit", default=None,
                    help="directory for score_run.py-shaped JSONs of the probe conditions")
    args = ap.parse_args()

    qrels = {d: load_qrels(os.path.join(args.root, "data", "track1_tempo", d), "dev")
             for d in DOMAINS}
    pools = {m: {d: load_pool(args.root, args.dense_tag, m, d) for d in DOMAINS}
             for m in args.models}

    def run_fused(m, weight):
        per = {}
        for d in DOMAINS:
            r = {}
            for q, (docs, sd, sce) in pools[m][d].items():
                s = z(sd) + weight * z(sce)
                order = [docs[i] for i in np.argsort(-s, kind="stable")]
                r[q] = ndcg_of(order, qrels[d].get(q, []))
            per[d] = r
        return per

    report = {"weights": args.weights, "models": {}, "transplant": {}, "overlap": {}}
    C = {}

    for m in args.models:
        rows = {"ce_only": {"macro": macro(run_fused(m, 1e9))},
                "dense_only": {"macro": macro(run_fused(m, 0.0))}}
        for w in args.weights:
            rows[f"w={w:g}"] = {"macro": macro(run_fused(m, w))}
        report["models"][m] = rows

    # rank-preserving transplant: ordering from `order_of`, z profile from `prof_of`
    a, b = args.models[0], args.models[1]

    def transplant(order_of, prof, weight):
        per = {}
        for d in DOMAINS:
            r = {}
            for q, (docs_o, _, sce_o) in pools[order_of][d].items():
                docs_p, sd_p, sce_p = pools[prof][d][q]
                zprof = dict(zip(docs_p, z(sce_p)))
                prof_sorted = np.sort([zprof[x] for x in docs_o])[::-1]
                rank = np.argsort(-sce_o, kind="stable")
                ce = np.empty(len(docs_o), dtype=float)
                ce[rank] = prof_sorted
                s = z(pools[order_of][d][q][1]) + weight * ce
                order = [docs_o[i] for i in np.argsort(-s, kind="stable")]
                r[q] = ndcg_of(order, qrels[d].get(q, []))
            per[d] = r
        return per

    for prof, order_of in ((a, b), (b, a)):
        for w in (1.0, 2.0):
            per = transplant(order_of, prof, w)
            label = f"order={order_of}|profile={prof}|w={w:g}"
            report["transplant"][label] = {"macro": macro(per)}
            C[label] = per

    # calibration-free control: reciprocal rank fusion over the same two lists
    for m in args.models:
        per = {}
        for d in DOMAINS:
            r = {}
            for q, (docs, sd, sce) in pools[m][d].items():
                rd = np.argsort(np.argsort(-sd, kind="stable"), kind="stable")
                rc = np.argsort(np.argsort(-sce, kind="stable"), kind="stable")
                s = 1.0 / (60 + rd) + 1.0 / (60 + rc)
                order = [docs[i] for i in np.argsort(-s, kind="stable")]
                r[q] = ndcg_of(order, qrels[d].get(q, []))
            per[d] = r
        report.setdefault("rrf", {})[m] = {"macro": macro(per)}
        C[f"{m}@rrf60"] = per

    # score-profile shape: mean z of the CE's own ranking position
    prof = {}
    for m in args.models:
        ranks = {r: [] for r in (1, 2, 3, 5, 10, 50)}
        for d in DOMAINS:
            for q, (docs, sd, sce) in pools[m][d].items():
                zz = np.sort(z(sce))[::-1]
                for r in ranks:
                    ranks[r].append(zz[r - 1])
        prof[m] = {f"z@rank{r}": float(np.mean(v)) for r, v in ranks.items()}
        prof[m]["z@rank1_minus_rank2"] = prof[m]["z@rank1"] - prof[m]["z@rank2"]
        prof[m]["z@rank1_minus_rank10"] = prof[m]["z@rank1"] - prof[m]["z@rank10"]
    report["profile"] = prof

    # how much the CE moves the fused top ten at w = 1
    for m in args.models:
        same_dense, same_ce, n = 0, 0, 0
        for d in DOMAINS:
            for q, (docs, sd, sce) in pools[m][d].items():
                s = z(sd) + z(sce)
                order = [docs[i] for i in np.argsort(-s, kind="stable")][:10]
                top_d = [docs[i] for i in np.argsort(-sd, kind="stable")][:10]
                top_c = [docs[i] for i in np.argsort(-sce, kind="stable")][:10]
                same_dense += len(set(order) & set(top_d))
                same_ce += len(set(order) & set(top_c))
                n += 1
        report["overlap"][m] = {"queries": n,
                                "fused_top10_vs_dense_top10": same_dense / (10 * n),
                                "fused_top10_vs_ce_top10": same_ce / (10 * n)}

    for m in args.models:
        for w in (0.75, 1.0, 1.25, 2.0):
            C[f"{m}@w={w:g}"] = run_fused(m, w)

    if args.emit:
        os.makedirs(args.emit, exist_ok=True)
        for cond, per in C.items():
            for d in DOMAINS:
                if not per.get(d):
                    continue
                rec = {"tag": cond, "data": os.path.join(args.root, "data", "track1_tempo", d),
                       "split": "dev", "num_topics": len(qrels[d]),
                       "per_query": {q: {"nDCG@10": v} for q, v in per[d].items()}}
                fn = f"score_{cond.replace('|', '_').replace('=', '').replace(' ', '')}_{d}_dev.json"
                with open(os.path.join(args.emit, fn), "w", encoding="utf-8") as f:
                    json.dump(rec, f)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for m, rows in report["models"].items():
        print(m, {k: round(v["macro"], 4) for k, v in rows.items()})
    for k, v in report["transplant"].items():
        print(k, round(v["macro"], 4))
    for m, v in report["profile"].items():
        print(m, {k: round(x, 3) for k, x in v.items()})
    print(json.dumps(report["overlap"], indent=1))


if __name__ == "__main__":
    main()
