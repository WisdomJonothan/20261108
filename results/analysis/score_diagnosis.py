#!/usr/bin/env python
"""SCORE_DIAGNOSIS: quantified diagnosis of the frozen dev score (domain-macro nDCG@10).

Analysis only. Reads committed artefacts, writes only under results/analysis/.
Official metric is re-implemented exactly as scripts/ir_metrics.py (binary attention):
  dcg(rels) = sum_i rel_i / log2(i+2),  idcg = dcg([1]*min(len(gold),10)), ndcg = dcg/idcg
Aggregation: mean nDCG@10 within domain, then equal weight across the 13 domains.
"""
import json, math, os, statistics
from collections import OrderedDict, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]
W = [1.0 / math.log2(i + 2) for i in range(10)]      # DCG position weights
R = lambda p: os.path.join(ROOT, p)


def dcg(rels):
    return sum(r * W[i] for i, r in enumerate(rels[:10]))


def ndcg10(ranked, gold):
    """ranked: iterable of docids in rank order; gold: set of docids."""
    if not gold:
        return 0.0
    ideal = dcg([1] * min(len(gold), 10))
    return dcg([1.0 if d in gold else 0.0 for d in list(ranked)[:10]]) / ideal


def read_qrels(domain):
    out = defaultdict(set)
    with open(R(f"data/track1_tempo/{domain}/qrels_dev.txt"), encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 4 and int(p[3]) > 0:
                out[p[0]].add(p[2])
    return out


def read_run(path):
    """-> OrderedDict qid -> [docids in rank order] (TREC rows sorted by score desc)."""
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 6:
                continue
            rows[p[0]].append((float(p[4]), int(p[3]), p[2]))
    out = OrderedDict()
    for q, rs in rows.items():
        rs.sort(key=lambda t: (-t[0], t[1]))
        out[q] = [t[2] for t in rs]
    return out


def load_json(p):
    with open(R(p), encoding="utf-8") as f:
        return json.load(f)


def domain_macro(per_query_scores, qids_by_domain):
    """per_query_scores: {qid: float}; equal weight per domain."""
    per = {}
    for d in DOMAINS:
        qs = qids_by_domain[d]
        per[d] = sum(per_query_scores.get(q, 0.0) for q in qs) / len(qs)
    return sum(per[d] for d in DOMAINS) / len(DOMAINS), per


# ---------------------------------------------------------------- load inputs
qrels = {d: read_qrels(d) for d in DOMAINS}
qids = {d: sorted(qrels[d]) for d in DOMAINS}
for d in DOMAINS:
    assert qids[d], d
N_TOTAL = sum(len(qids[d]) for d in DOMAINS)

score_files = {d: load_json(f"results/score_sub_sft_e2bc_{d}_dev.json") for d in DOMAINS}
frozen = {d: {q: v["nDCG@10"] for q, v in score_files[d]["per_query"].items()} for d in DOMAINS}
base_macro, base_per_domain = domain_macro(
    {q: s for d in DOMAINS for q, s in frozen[d].items()}, qids)

# independent recomputation from the committed head-10 run files
head10 = {d: read_run(f"runs/sub_sft_e2bc_{d}_dev_head10.trec") for d in DOMAINS}
frozen_re = {}
for d in DOMAINS:
    for q in qids[d]:
        frozen_re[q] = ndcg10(head10[d].get(q, [])[:10], qrels[d][q])
re_macro, re_per = domain_macro(frozen_re, qids)
assert abs(re_macro - base_macro) < 1e-12, (re_macro, base_macro)
max_dev = max(abs(frozen_re[q] - frozen[d][q]) for d in DOMAINS for q in qids[d])

# ---------------------------------------------------------------- 1. distribution
all_q = [(d, q, frozen[d][q]) for d in DOMAINS for q in qids[d]]
vals = [v for _, _, v in all_q]
zeros = [(d, q) for d, q, v in all_q if v == 0.0]
nonzero = [v for v in vals if v > 0]
hist = OrderedDict((f"{i/10:.1f}-{(i+1)/10:.1f}", 0) for i in range(10))
for v in vals:
    hist[f"{min(int(v*10),9)/10:.1f}-{(min(int(v*10),9)+1)/10:.1f}"] += 1
deciles = [{"lo": i / 10, "hi": (i + 1) / 10,
            "n": sum(1 for v in vals if i / 10 <= v < (i + 1) / 10)} for i in range(9)]
deciles.append({"lo": 0.9, "hi": 1.0, "n": sum(1 for v in vals if v >= 0.9)})

# counterfactuals for the zero class
med_all = {d: statistics.median([frozen[d][q] for q in qids[d]]) for d in DOMAINS}
mean_all = {d: base_per_domain[d] for d in DOMAINS}
nz = {d: [frozen[d][q] for q in qids[d] if frozen[d][q] > 0] for d in DOMAINS}
med_nz = {d: statistics.median(nz[d]) for d in DOMAINS}
mean_nz = {d: sum(nz[d]) / len(nz[d]) for d in DOMAINS}
n_zero = {d: sum(1 for q in qids[d] if frozen[d][q] == 0.0) for d in DOMAINS}


def lift_cf(fill):
    s, per = 0.0, {}
    for d in DOMAINS:
        num = sum(fill[d] if frozen[d][q] == 0.0 else frozen[d][q] for q in qids[d])
        per[d] = num / len(qids[d])
        s += per[d]
    return s / 13, per


cf_med, cf_med_per = lift_cf(med_all)
cf_mednz, cf_mednz_per = lift_cf(med_nz)
cf_meannz, cf_meannz_per = lift_cf(mean_nz)
# "locked points": per-domain contribution of the zero class under each fill
locked = OrderedDict()
for d in DOMAINS:
    locked[d] = {
        "n": len(qids[d]), "n_zero": n_zero[d], "share_zero": n_zero[d] / len(qids[d]),
        "domain_median": med_all[d], "nonzero_mean": mean_nz[d],
        "points_at_domain_median": (1 / 13) * n_zero[d] / len(qids[d]) * med_all[d],
        "points_at_nonzero_mean": (1 / 13) * n_zero[d] / len(qids[d]) * mean_nz[d],
    }

# ---------------------------------------------------------------- 2/3. per domain + answerability
dense = {d: read_run(f"runs/dense_mxbai_{d}_dev.trec") for d in DOMAINS}
dense_ndcg = {}
for d in DOMAINS:
    for q in qids[d]:
        dense_ndcg[q] = ndcg10(dense[d].get(q, [])[:10], qrels[d][q])
dense_macro, dense_per = domain_macro(dense_ndcg, qids)

per_domain = OrderedDict()
for d in DOMAINS:
    n = len(qids[d])
    oracle100 = sum(score_files[d]["per_query"][q]["oracle@10_cand100"] for q in qids[d]) / n
    oracle_all = sum(score_files[d]["per_query"][q]["oracle@10_cand_all"] for q in qids[d]) / n
    rec100 = sum(score_files[d]["per_query"][q]["Rec@100"] for q in qids[d]) / n
    rec1000 = sum(score_files[d]["per_query"][q]["Rec@1000"] for q in qids[d]) / n
    answerable = sum(1 for q in qids[d] if len(qrels[d][q]) > 0)
    in_dense1000 = sum(1 for q in qids[d] if any(x in qrels[d][q] for x in dense[d].get(q, [])))
    per_domain[d] = {
        "n": n, "weight": 1 / 13, "n_zero": n_zero[d], "zero_share": n_zero[d] / n,
        "baseline": base_per_domain[d], "median": med_all[d], "nonzero_mean": mean_nz[d],
        "dense_only": dense_per[d],
        "oracle_cand100": oracle100, "oracle_cand_all": oracle_all,
        "headroom_to_oracle100": oracle100 - base_per_domain[d],
        "recall100": rec100, "recall1000": rec1000,
        "answerable_queries": answerable,
        "queries_with_gold_in_dense1000": in_dense1000,
        "points_contribution": base_per_domain[d] / 13,
    }

n_answerable = sum(per_domain[d]["answerable_queries"] for d in DOMAINS)
perfect_macro = sum(per_domain[d]["answerable_queries"] / per_domain[d]["n"] for d in DOMAINS) / 13
reach_macro = sum(per_domain[d]["queries_with_gold_in_dense1000"] / per_domain[d]["n"] for d in DOMAINS) / 13

# ---------------------------------------------------------------- 4. depth curves
def oracle_by_depth(run, domain, depth):
    """ideal top-10 from the first `depth` docs of the run; official qrels IDCG."""
    out = {}
    for q in qids[d]:
        docs = run[domain].get(q, [])[:depth]
        p = sum(1 for x in docs if x in qrels[domain][q])
        out[q] = dcg([1.0] * min(p, 10)) / dcg([1.0] * min(len(qrels[domain][q]), 10))
    return out


CE_CANDIDATES = ["rerank_sft_e2b", "rerank_lx_ep1b", "rerank_sft_e2s29",
                 "fusion100_full_ce_s17_20261115", "rerank_cx_bf"]
ce_actual = {}
for tag in CE_CANDIDATES:
    ok, per = True, {}
    for d in DOMAINS:
        path = f"runs/{tag}_{d}_dev.trec"
        if not os.path.exists(R(path)):
            path = f"runs/{tag}_{d}_dev_ce.trec"
            if not os.path.exists(R(path)):
                ok = False
                break
        run = read_run(path)
        per[d] = {q: ndcg10(run.get(q, [])[:10], qrels[d][q]) for q in qids[d]}
    if ok:
        m, _ = domain_macro({q: s for d in DOMAINS for q, s in per[d].items()}, qids)
        ce_actual[tag] = {"ndcg10": m, "per_domain": {d: sum(per[d].values()) / len(per[d]) for d in DOMAINS}}

depth_curve = OrderedDict()
for depth in (10, 20, 50, 100, 200, 300, 500, 1000):
    pq = {}
    for d in DOMAINS:
        pq.update(oracle_by_depth(dense, d, depth))
    m, per = domain_macro(pq, qids)
    depth_curve[f"dense@{depth}"] = {"macro": m, "per_domain": per}

CE_TAG = "rerank_sft_e2b"
ce_run = {}
for d in DOMAINS:
    p = f"runs/{CE_TAG}_{d}_dev.trec"
    ce_run[d] = read_run(p) if os.path.exists(R(p)) else {}
for depth in (10, 20, 50, 100):
    pq = {}
    for d in DOMAINS:
        pq.update(oracle_by_depth(ce_run, d, depth))
    m, per = domain_macro(pq, qids)
    depth_curve[f"ce@{depth}"] = {"macro": m, "per_domain": per}

# depth at which the 138 zero queries become reachable (dense top-1000)
zero_depth = OrderedDict()
zr = load_json("results/replacement_v2/zero_recall_dev.json")
for rec in zr["per_query"]:
    pass
zero_gold_probe = []
for d, q in zeros:
    docs = dense[d].get(q, [])
    best = next((i + 1 for i, x in enumerate(docs) if x in qrels[d][q]), None)
    zero_gold_probe.append({"domain": d, "qid": q, "best_dense_rank": best,
                            "n_gold": len(qrels[d][q])})
ranks = [z["best_dense_rank"] for z in zero_gold_probe]
zero_depth = {
    "n_zero": len(ranks),
    "none_in_dense1000": sum(1 for r in ranks if r is None),
    "le20": sum(1 for r in ranks if r and r <= 20),
    "le50": sum(1 for r in ranks if r and r <= 50),
    "le100": sum(1 for r in ranks if r and r <= 100),
    "le200": sum(1 for r in ranks if r and r <= 200),
    "le1000": sum(1 for r in ranks if r),
}

# ---------------------------------------------------------------- 5. swap ladders (closed form)
# Replacing slot i by a relevant pool doc gains (1-rel_i)*w_i. With p relevant pool docs
# available, the ideal k-swap places them at the k largest-weight currently-wrong slots.
def ladder(base_scores, pool_of, slots=None, ks=(1, 2, 3)):
    """Returns macro after ideal replacement of up to k slots, for each k."""
    res = {}
    for k in ks:
        pq = {}
        for d in DOMAINS:
            for q in qids[d]:
                gold = qrels[d][q]
                top = head10[d].get(q, [])[:10]
                rel = [1.0 if x in gold else 0.0 for x in top]
                pool = [x for x in pool_of(d, q) if x not in set(top)]
                p = sum(1 for x in pool if x in gold)
                idx = [i for i in range(len(top)) if rel[i] == 0.0 and (slots is None or i in slots)]
                idx.sort(key=lambda i: -W[i])
                take = min(p, k, len(idx))
                gain = sum(W[i] for i in idx[:take])
                pq[q] = base_scores[q] + gain / dcg([1.0] * min(len(gold), 10))
        m, per = domain_macro(pq, qids)
        res[k] = {"macro": m, "per_domain": per}
    return res


def pool_dense(limit):
    return lambda d, q: dense[d].get(q, [])[:limit]


ladder_dense100 = ladder(frozen_re, pool_dense(100), ks=(1, 2, 3, 10))
ladder_dense20 = ladder(frozen_re, pool_dense(20), ks=(1, 2, 3))
ladder_dense100_band = {
    "1-3": ladder(frozen_re, pool_dense(100), slots=set(range(0, 3)), ks=(1, 2, 3)),
    "4-7": ladder(frozen_re, pool_dense(100), slots=set(range(3, 7)), ks=(1, 2, 3)),
    "8-10": ladder(frozen_re, pool_dense(100), slots=set(range(7, 10)), ks=(1, 2, 3)),
}
ce100 = {d: ce_run[d] for d in DOMAINS}
ladder_ce100 = None
if all(ce100.values()):
    pool_ce = lambda d, q: ce100[d].get(q, [])[:100]
    ladder_ce100 = ladder(frozen_re, pool_ce, ks=(1, 2, 3))

# ---------------------------------------------------------------- write JSON
out = {
    "meta": {
        "frozen_baseline_macro": base_macro,
        "recomputed_from_head10": re_macro,
        "max_abs_per_query_diff_vs_score_files": max_dev,
        "n_queries": N_TOTAL,
        "n_domains": 13,
        "metric": "binary nDCG@10, mean within domain, equal weight across 13 domains",
        "sources": {
            "per_query_baseline": "results/score_sub_sft_e2bc_<domain>_dev.json:per_query",
            "frozen_runs": "runs/sub_sft_e2bc_<domain>_dev_head10.trec",
            "dense": "runs/dense_mxbai_<domain>_dev.trec (depth 1000)",
            "ce_only": f"runs/{CE_TAG}_<domain>_dev.trec (depth 100)",
            "qrels": "data/track1_tempo/<domain>/qrels_dev.txt",
        },
    },
    "distribution": {
        "n": len(vals), "n_zero": len(zeros), "zero_share": len(zeros) / len(vals),
        "mean": sum(vals) / len(vals), "median": statistics.median(vals),
        "mean_nonzero": sum(nonzero) / len(nonzero), "median_nonzero": statistics.median(nonzero),
        "n_nonzero": len(nonzero), "n_one": sum(1 for v in vals if v >= 1.0),
        "deciles": deciles, "hist": hist,
    },
    "zero_class": {
        "n_zero": len(zeros),
        "macro_if_zeros_at_domain_median": cf_med,
        "macro_if_zeros_at_domain_nonzero_median": cf_mednz,
        "macro_if_zeros_at_domain_nonzero_mean": cf_meannz,
        "delta_median_points": (cf_med - base_macro) * 100,
        "delta_nonzero_mean_points": (cf_meannz - base_macro) * 100,
        "per_domain": locked,
        "zero_depth_in_dense": zero_depth,
        "zero_probe": zero_gold_probe,
    },
    "per_domain": per_domain,
    "answerability": {
        "n_answerable": n_answerable, "n_unanswerable": N_TOTAL - n_answerable,
        "theoretical_perfect_macro": perfect_macro,
        "macro_if_only_dense1000_reachable_gold": reach_macro,
        "note": "a query is answerable iff its qrels_dev has >=1 positive doc; "
                "the corpus ceiling assumes every such doc can be ranked first.",
    },
    "depth_curve": depth_curve,
    "dense_actual_macro": dense_macro,
    "ce_actual": ce_actual,
    "ce_depth_tag": CE_TAG,
    "ladders": {
        "dense_top100": ladder_dense100,
        "dense_top20": ladder_dense20,
        "dense_top100_by_band": ladder_dense100_band,
        "ce_top100": ladder_ce100,
    },
}
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "SCORE_DIAGNOSIS.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- console
def pct(x):
    return f"{100*x:.2f}%"


print(f"queries={N_TOTAL} baseline={base_macro:.10f} (recomputed {re_macro:.10f}, max diff {max_dev:.2e})")
print(f"zeros={len(zeros)} ({pct(len(zeros)/len(vals))}) mean_nonzero={sum(nonzero)/len(nonzero):.4f} "
      f"median_nz={statistics.median(nonzero):.4f} n_at_1={sum(1 for v in vals if v>=1)}")
print("deciles: " + " ".join(f"{d['lo']:.1f}:{d['n']}" for d in deciles))
print(f"zero->domain median macro {cf_med:.4f} (+{100*(cf_med-base_macro):.2f} pts); "
      f"zero->nonzero mean {cf_meannz:.4f} (+{100*(cf_meannz-base_macro):.2f}); "
      f"zero->nonzero median {cf_mednz:.4f} (+{100*(cf_mednz-base_macro):.2f})")
print("\nper-domain: n zero% base dense oracle100 headroom rec100 rec1000")
for d in DOMAINS:
    x = per_domain[d]
    print(f"  {d:10s} {x['n']:3d} {pct(x['zero_share']):>7s} {x['baseline']:.4f} {x['dense_only']:.4f} "
          f"{x['oracle_cand100']:.4f} {x['headroom_to_oracle100']:+.4f} {x['recall100']:.4f} {x['recall1000']:.4f}")
print(f"\nanswerable={n_answerable}/{N_TOTAL} unanswerable={N_TOTAL-n_answerable} "
      f"perfect_macro={perfect_macro:.4f} dense1000_reachable_macro={reach_macro:.4f}")
print("\ndepth curve (oracle nDCG@10 of the pool):")
for k, v in depth_curve.items():
    print(f"  {k:12s} {v['macro']:.4f}")
print("\nCE-only actual nDCG@10:", {k: round(v["ndcg10"], 5) for k, v in ce_actual.items()})
print(f"dense-only actual nDCG@10: {dense_macro:.4f}")
print("\nladders (ideal k-slot replacement):")
for name, lad in (("dense@100", ladder_dense100), ("dense@20", ladder_dense20),
                  ("ce@100", ladder_ce100)):
    if lad:
        print(f"  {name}: " + " ".join(f"k={k}:{v['macro']:.4f}" for k, v in lad.items()))
for band, lad in ladder_dense100_band.items():
    print(f"  dense@100 band {band}: " + " ".join(f"k={k}:{v['macro']:.4f}" for k, v in lad.items()))
print("\nzero-class dense best rank:", zero_depth)
