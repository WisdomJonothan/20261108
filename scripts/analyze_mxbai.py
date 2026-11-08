#!/usr/bin/env python3
"""Dense (mxbai) vs cross-encoder rerank vs RRF fusion, per domain + pooled.

Same protocol as analyze_phase7.py, but the runs are
  runs/dense_mxbai_<domain>_dev.trec
  runs/rerank_mxbai_<domain>_dev.trec
so the stronger first-stage retriever can be compared against the bge-base
pipeline under identical weights.

Outputs JSON with per-domain metrics, pooled query-bootstrap deltas,
domain-clustered bootstrap and leave-one-domain-out weight selection.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_qrels, read_trec_run  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAINS = "bitcoin economics politics genealogy hsm history law workplace cardano monero quant iota travel".split()


def ndcg_at(order, gold, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(order[:k]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / idcg if idcg else 0.0


def rrf(a, b, wa, wb, c=60.0):
    sa = {d: i + 1 for i, d in enumerate(a)}
    sb = {d: i + 1 for i, d in enumerate(b)}
    score = {}
    for d in sa:
        s = wa / (c + sa[d])
        if d in sb:
            s += wb / (c + sb[d])
        score[d] = s
    for d in sb:
        if d not in sa:
            score[d] = wb / (c + sb[d])
    return sorted(score, key=lambda d: (-score[d], d))


def boot(xs, n=5000, seed=0):
    if not xs:
        return None
    rng = random.Random(seed)
    m = len(xs)
    bs = sorted(sum(xs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {"n": m, "mean": statistics.mean(xs),
            "ci": [bs[125], bs[4875]],
            "p_gt0": sum(1 for v in bs if v > 0) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "phase8_mxbai_summary.json"))
    ap.add_argument("--fusions", default="2,1;1,1;1,2;1,4")
    args = ap.parse_args()
    weights = [tuple(float(x) for x in s.split(",")) for s in args.fusions.split(";")]

    rows, pooled, per_q = [], {f"fused_{a:g}:{b:g}": [] for a, b in weights}, {}
    pooled_ce = []
    for dom in DOMAINS:
        ddir = os.path.join(ROOT, "data", "track1_tempo", dom)
        dense_p = os.path.join(ROOT, "runs", f"dense_mxbai_{dom}_dev.trec")
        ce_p = os.path.join(ROOT, "runs", f"rerank_mxbai_{dom}_dev.trec")
        if not os.path.isfile(dense_p):
            continue
        D = read_trec_run(dense_p)
        CE = read_trec_run(ce_p) if os.path.isfile(ce_p) else None
        qrels = load_qrels(ddir, "dev")
        base = {q: ndcg_at([t[2] for t in D[q]], set(qrels.get(q, {}))) for q in D if q in qrels}
        row = {"domain": dom, "n": len(base), "dense": statistics.mean(base.values()) if base else 0,
               "ce": None, "fusions": {}}
        if CE:
            ce_vals = {q: ndcg_at([t[2] for t in CE[q]], set(qrels.get(q, {})))
                       for q in base if q in CE}
            row["ce"] = statistics.mean(ce_vals.values())
            pooled_ce += [ce_vals[q] - base[q] for q in ce_vals]
            for wa, wb in weights:
                key = f"fused_{wa:g}:{wb:g}"
                vals = {}
                for q in base:
                    if q not in CE:
                        continue
                    a = [t[2] for t in D[q]]
                    b = [t[2] for t in CE[q]]
                    vals[q] = ndcg_at(rrf(a, b, wa, wb), set(qrels.get(q, {})))
                row["fusions"][key] = statistics.mean(vals.values())
                pooled[key] += [vals[q] - base[q] for q in vals]
        rows.append(row)

    summary = {"domains": rows,
               "macro": {"dense": statistics.mean([r["dense"] for r in rows]),
                         "ce": statistics.mean([r["ce"] for r in rows if r["ce"] is not None])},
               "pooled": {"ce": boot(pooled_ce),
                          **{k: boot(v) for k, v in pooled.items()}}}
    for wa, wb in weights:
        summary["macro"][f"fused_{wa:g}:{wb:g}"] = statistics.mean(
            [r["fusions"][f"fused_{wa:g}:{wb:g}"] for r in rows if r["fusions"]])

    # domain-clustered bootstrap of the dense->fused deltas
    dommat = []
    for r in rows:
        if r["fusions"]:
            dommat.append([r["fusions"][k] - r["dense"] for k in r["fusions"]])
    keys = [k for k in rows[-1]["fusions"]] if rows and rows[-1]["fusions"] else []
    clust = {}
    if dommat:
        rng = random.Random(1)
        nd = len(dommat)
        for j, key in enumerate(keys):
            means = []
            for _ in range(3000):
                idx = [rng.randrange(nd) for _ in range(nd)]
                means.append(statistics.mean(dommat[i][j] for i in idx))
            means.sort()
            clust[key] = {"mean": statistics.mean(d[j] for d in dommat),
                          "ci": [means[75], means[2924]],
                          "domains": nd}

    # leave-one-domain-out weight selection
    loo = {"folds": [], "mean_holdout": None}
    if dommat and keys:
        picks = []
        for i in range(len(dommat)):
            best, bestv = None, -1e9
            for j, key in enumerate(keys):
                v = statistics.mean(dommat[t][j] for t in range(len(dommat)) if t != i)
                if v > bestv:
                    best, bestv = key, v
            hold = dommat[i][keys.index(best)]
            picks.append({"fold_domain": rows[i]["domain"], "picked": best, "heldout_gain": hold})
        loo = {"folds": picks, "mean_holdout": statistics.mean(p["heldout_gain"] for p in picks)}

    summary["domain_clustered"] = clust
    summary["loocv"] = loo
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    for r in rows:
        print(f'{r["domain"]:10s} n={r["n"]:3d} dense {r["dense"]:.4f} '
              f'ce {("-" if r["ce"] is None else f"{r["ce"]:.4f}")} ' +
              " ".join(f'{k.replace("fused_","")} {v:.4f}' for k, v in r["fusions"].items()))
    print(json.dumps({"macro": {k: round(v, 4) for k, v in summary["macro"].items()},
                      "pooled_ce": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary["pooled"]["ce"].items()},
                      "clustered": {k: {"mean": round(v["mean"], 4), "ci": [round(x, 4) for x in v["ci"]]}
                                    for k, v in clust.items()},
                      "loocv": {"mean_holdout": round(loo["mean_holdout"], 4) if loo["mean_holdout"] is not None else None}}, indent=1))


if __name__ == "__main__":
    main()
