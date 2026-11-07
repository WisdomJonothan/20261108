"""Phase-7 analysis: dense vs cross-encoder vs RRF fusion, all Track-1a domains.

Everything is recomputed from the TREC run files + qrels, so alternate fusion
weights can be evaluated without re-running retrieval.

For each domain (dev split):
  dense           : dense_bge_<dom>_dev.trec
  CE              : rerank_v2m3_(q)title_<dom>_dev.trec
  fused(wa,wb)    : RRF(c=60) over dense and CE

Reports per-domain nDCG@10 and query-paired deltas vs dense, plus pooled
estimates across all domains with a query-level bootstrap CI.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, r"D:\RETECO\scripts")
from lib_reteco import load_qrels, read_trec_run  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAINS = ["bitcoin", "economics", "iota", "cardano", "genealogy", "history",
           "hsm", "law", "monero", "politics", "quant", "travel", "workplace"]


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


def run_for(domain, kind):
    d = os.path.join(ROOT, "runs")
    if kind == "dense":
        return os.path.join(d, f"dense_bge_{domain}_dev.trec")
    for name in (f"rerank_v2m3_qtitle_{domain}_dev.trec",
                 f"rerank_v2m3_title_{domain}_dev.trec"):
        if os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "results", "phase7_summary.json"))
    ap.add_argument("--fusions", default="2,1;1,2;1,1")
    args = ap.parse_args()
    weights = [tuple(float(x) for x in spec.split(",")) for spec in args.fusions.split(";")]

    qrels_dir = os.path.join(ROOT, "data", "track1_tempo")
    rows, pooled = [], {f"fused_{wa:g}:{wb:g}": [] for wa, wb in weights}
    pooled_ce = []
    for dom in DOMAINS:
        dense_p = run_for(dom, "dense")
        ce_p = run_for(dom, "ce")
        if not (dense_p and os.path.isfile(dense_p)):
            continue
        qrels = load_qrels(os.path.join(qrels_dir, dom), "dev")
        D = read_trec_run(dense_p)
        CE = read_trec_run(ce_p) if ce_p else None
        base = {q: ndcg_at([t[2] for t in D[q]], set(qrels.get(q, {}))) for q in D if q in qrels}
        row = {"domain": dom, "n": len(base), "dense": statistics.mean(base.values()),
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

    def stats(xs):
        if not xs:
            return None
        rng = random.Random(0)
        n = len(xs)
        bs = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(5000))
        return {"n": n, "mean": statistics.mean(xs),
                "ci": [bs[125], bs[4875]],
                "p_gt0": sum(1 for v in bs if v > 0) / len(bs)}


    # leave-one-domain-out selection of the fusion weight (dev-level protocol)
    if len([r for r in rows if r["ce"] is not None]) >= 3:
        grid = list(weights)
        loo = {}
        for hold in [r for r in rows if r["ce"] is not None]:
            best, best_val = None, -1e9
            for wa, wb in grid:
                vals = []
                for r in rows:
                    if r is hold or r["ce"] is None:
                        continue
                    vals.append(r["fusions"].get(f"fused_{wa:g}:{wb:g}") - r["dense"])
                if vals and statistics.mean(vals) > best_val:
                    best_val = statistics.mean(vals)
                    best = (wa, wb)
            key = f"fused_{best[0]:g}:{best[1]:g}"
            loo[hold["domain"]] = {"chosen": key,
                                   "gain_on_holdout": hold["fusions"][key] - hold["dense"]}
        LOO = {"selection": loo,
               "mean_gain": statistics.mean(v["gain_on_holdout"] for v in loo.values())}

    out = {"domains": rows,
           "pooled_ce_vs_dense": stats(pooled_ce),
           "pooled_fusions": {k: stats(v) for k, v in pooled.items()}}
    if "LOO" in dir():
        out["loocv"] = LOO
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    hdr = f"{'domain':12}{'n':>5}{'dense':>8}{'CE':>8}"
    for wa, wb in weights:
        hdr += f"{f'R{wa:g}:{wb:g}':>8}"
    print(hdr)
    for r in rows:
        line = f"{r['domain']:12}{r['n']:>5}{r['dense']:>8.4f}"
        line += f"{r['ce']:>8.4f}" if r["ce"] is not None else f"{'-':>8}"
        for wa, wb in weights:
            v = r["fusions"].get(f"fused_{wa:g}:{wb:g}")
            line += f"{v:>8.4f}" if v is not None else f"{'-':>8}"
        print(line)
    print("pooled CE vs dense:", out["pooled_ce_vs_dense"])
    for k, v in out["pooled_fusions"].items():
        print("pooled", k, v)


if __name__ == "__main__":
    main()
