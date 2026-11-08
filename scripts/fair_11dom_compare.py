#!/usr/bin/env python3
"""Fair bge-base vs mxbai comparison on the shared 11 dev domains.

The two published pipelines cover different 12-domain sets:
  old: 12 domains = 11 shared + travel (no iota)
  new: 12 domains = 11 shared + iota  (no travel)
Iota has 3 dev queries, so its inclusion can inflate a macro average. This
script recomputes both pipelines on the *shared* 11 domains only.

RRF is recomputed here (c=60) from the dense and CE runs, so the weights are
exactly the published ones (bge 1:4, mxbai 1:1).
"""
import json
import os

import numpy as np

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAINS = ["bitcoin", "economics", "politics", "genealogy", "hsm", "history",
           "law", "workplace", "cardano", "monero", "quant"]


def read_run(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) < 6:
            continue
        out.setdefault(p[0], []).append((int(p[3]), p[2]))
    for q in out:
        out[q].sort()
        out[q] = [d for _r, d in out[q]]
    return out


def read_qrels(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 4:
            out.setdefault(p[0], set()).add(p[2])
    return out


def ndcg(ranked, gold, k=10):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def rrf(a, b, wa, wb, c=60.0):
    out = {}
    for q in set(a) | set(b):
        ra = {d: i + 1 for i, d in enumerate(a.get(q, []))}
        rb = {d: i + 1 for i, d in enumerate(b.get(q, []))}
        s = {d: wa / (c + ra.get(d, 10000)) + wb / (c + rb.get(d, 10000))
             for d in set(ra) | set(rb)}
        out[q] = [d for d, _ in sorted(s.items(), key=lambda t: -t[1])]
    return out


def per_query(run, qrels):
    return {q: ndcg(run.get(q, []), gold) for q, gold in qrels.items() if gold}


def main():
    systems = {
        "bge_dense": {}, "bge_rrf_1:4": {}, "mxbai_dense": {}, "mxbai_rrf_1:1": {},
    }
    for d in DOMAINS:
        base = os.path.join(ROOT, "data", "track1_tempo", d)
        qrels = read_qrels(os.path.join(base, "qrels_dev.txt"))
        bd = read_run(os.path.join(ROOT, "runs", f"dense_bge_{d}_dev.trec"))
        bce = os.path.join(ROOT, "runs", f"rerank_v2m3_title_{d}_dev.trec")
        if not os.path.isfile(bce):
            bce = os.path.join(ROOT, "runs", f"rerank_v2m3_qtitle_{d}_dev.trec")
        bc = read_run(bce)
        md = read_run(os.path.join(ROOT, "runs", f"dense_mxbai_{d}_dev.trec"))
        mc = read_run(os.path.join(ROOT, "runs", f"rerank_mxbai_{d}_dev.trec"))
        systems["bge_dense"].update(per_query(bd, qrels))
        systems["bge_rrf_1:4"].update(per_query(rrf(bd, bc, 1, 4), qrels))
        systems["mxbai_dense"].update(per_query(md, qrels))
        systems["mxbai_rrf_1:1"].update(per_query(rrf(md, mc, 1, 1), qrels))

    qs = sorted(systems["bge_dense"])
    dom_of = {}
    for d_ in DOMAINS:
        for q in read_qrels(os.path.join(ROOT, "data", "track1_tempo", d_, "qrels_dev.txt")):
            dom_of[q] = d_
    out = {"n_queries": len(qs), "domains": DOMAINS, "macro": {}, "delta_vs_bge_rrf": {}}
    for name, vals in systems.items():
        out["macro"][name] = round(float(np.mean([vals[q] for q in qs])), 4)
        dm = [np.mean([vals[q] for q in qs if dom_of[q] == d_]) for d_ in DOMAINS]
        out.setdefault("domain_macro", {})[name] = round(float(np.mean(dm)), 4)
    rng = np.random.default_rng(0)
    base = np.asarray([systems["bge_rrf_1:4"][q] for q in qs])
    for name, vals in systems.items():
        d = np.asarray([vals[q] for q in qs]) - base
        boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(5000)]
        out["delta_vs_bge_rrf"][name] = {
            "mean": round(float(d.mean()), 4),
            "ci": [round(float(np.percentile(boots, 2.5)), 4),
                   round(float(np.percentile(boots, 97.5)), 4)],
        }
    # domain-clustered delta: mxbai RRF vs bge RRF, and mxbai RRF vs mxbai dense
    def clustered(a_vals, b_vals):
        diffs = {d_: [] for d_ in DOMAINS}
        for q in qs:
            diffs[dom_of[q]].append(a_vals[q] - b_vals[q])
        dm = {d_: float(np.mean(v)) for d_, v in diffs.items() if v}
        boots = []
        for _ in range(5000):
            pick = rng.choice(DOMAINS, size=len(DOMAINS), replace=True)
            boots.append(float(np.mean([dm[p] for p in pick])))
        return round(float(np.mean(list(dm.values()))), 4), [
            round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4)]
    out["clustered"] = {
        "mxbai_rrf_vs_bge_rrf": clustered(systems["mxbai_rrf_1:1"], systems["bge_rrf_1:4"]),
        "mxbai_rrf_vs_mxbai_dense": clustered(systems["mxbai_rrf_1:1"], systems["mxbai_dense"]),
        "bge_rrf_vs_bge_dense": clustered(systems["bge_rrf_1:4"], systems["bge_dense"]),
    }
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "fair_11dom_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
