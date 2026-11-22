#!/usr/bin/env python
"""SCORE_DIAGNOSIS part 2: ordering-only ceilings, zero-class decomposition, union ladder,
CE-only depth curve, per-domain policy specialisation. Writes results/analysis/SCORE_DIAGNOSIS_EXTRA.json
"""
import json, math, os, statistics
from collections import OrderedDict, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]
W = [1.0 / math.log2(i + 2) for i in range(10)]
R = lambda p: os.path.join(ROOT, p)
dcg = lambda rels: sum(r * W[i] for i, r in enumerate(rels[:10]))


def ndcg10(ranked, gold):
    if not gold:
        return 0.0
    return dcg([1.0 if d in gold else 0.0 for d in list(ranked)[:10]]) / dcg([1.0] * min(len(gold), 10))


def qrels(d):
    o = defaultdict(set)
    with open(R(f"data/track1_tempo/{d}/qrels_dev.txt"), encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 4 and int(p[3]) > 0:
                o[p[0]].add(p[2])
    return o


def run(p):
    r = defaultdict(list)
    with open(R(p), encoding="utf-8") as f:
        for ln in f:
            s = ln.split()
            if len(s) >= 6:
                r[s[0]].append((float(s[4]), int(s[3]), s[2]))
    for q in r:
        r[q].sort(key=lambda t: (-t[0], t[1]))
    return {q: [t[2] for t in v] for q, v in r.items()}


def load(p):
    with open(R(p), encoding="utf-8") as f:
        return json.load(f)


Q = {d: qrels(d) for d in DOMAINS}
QIDS = {d: sorted(Q[d]) for d in DOMAINS}
DENSE = {d: run(f"runs/dense_mxbai_{d}_dev.trec") for d in DOMAINS}
FROZEN = {d: run(f"runs/sub_sft_e2bc_{d}_dev_head10.trec") for d in DOMAINS}
SUB = {d: load(f"results/score_sub_sft_e2bc_{d}_dev.json")["per_query"] for d in DOMAINS}


def macro(pq):
    per = {d: sum(pq[q] for q in QIDS[d]) / len(QIDS[d]) for d in DOMAINS}
    return sum(per.values()) / 13, per


base = {q: SUB[d][q]["nDCG@10"] for d in DOMAINS for q in QIDS[d]}
BASE, base_per = macro(base)

# ---- ordering-only ceiling: ideal order of the very same top-10 documents
reorder = {}
n_reorderable = 0
for d in DOMAINS:
    for q in QIDS[d]:
        gold = Q[d][q]
        top = FROZEN[d].get(q, [])[:10]
        rels = [1.0 if x in gold else 0.0 for x in top]
        reorder[q] = dcg(sorted(rels, reverse=True)) / dcg([1.0] * min(len(gold), 10))
        if reorder[q] > base[q] + 1e-12:
            n_reorderable += 1
REORDER, reorder_per = macro(reorder)

# ---- pool-oracle decomposition: zero class vs non-zero class
pooloracle = {q: SUB[d][q]["oracle@10_cand100"] for d in DOMAINS for q in QIDS[d]}
POOL, pool_per = macro(pooloracle)
zeros = OrderedDict()
for d in DOMAINS:
    zc = [q for q in QIDS[d] if base[q] == 0.0]
    zeros[d] = {
        "n": len(zc),
        "n_with_pool_gold": sum(1 for q in zc if pooloracle[q] > 0),
        "oracle_points_if_fixed": sum(pooloracle[q] for q in zc) / len(QIDS[d]) / 13,
        "median_of_domain": statistics.median([base[q] for q in QIDS[d]]),
        "nonzero_mean": (lambda v: sum(v) / len(v))([base[q] for q in QIDS[d] if base[q] > 0]),
    }
    zeros[d]["median_points_if_fixed"] = (1 / 13) * len(zc) / len(QIDS[d]) * zeros[d]["median_of_domain"]
    zeros[d]["nonzero_mean_points_if_fixed"] = (1 / 13) * len(zc) / len(QIDS[d]) * zeros[d]["nonzero_mean"]
ZERO_ORACLE_POINTS = sum(v["oracle_points_if_fixed"] for v in zeros.values())
ZERO_MEDIAN_POINTS = sum(v["median_points_if_fixed"] for v in zeros.values())
ZERO_NZMEAN_POINTS = sum(v["nonzero_mean_points_if_fixed"] for v in zeros.values())
ZERO_ORACLE_MACRO = BASE + ZERO_ORACLE_POINTS
nzero = sum(v["n"] for v in zeros.values())
nzero_pool = sum(v["n_with_pool_gold"] for v in zeros.values())

# ---- union(dense top20, CE top20) ladder, to verify the committed 0.5104 / 0.4098 rungs
CE_RUNS = {}
for tag, suffix in (("s17full", "fusion100_full_ce_s17_20261115_{d}_dev_ce.trec"),
                    ("sft_e2b", "rerank_sft_e2b_{d}_dev.trec")):
    ok = all(os.path.exists(R("runs/" + suffix.format(d=d))) for d in DOMAINS)
    if ok:
        CE_RUNS[tag] = {d: run("runs/" + suffix.format(d=d)) for d in DOMAINS}


def ladder_union(ce, k, slots=None):
    pq = {}
    for d in DOMAINS:
        for q in QIDS[d]:
            gold = Q[d][q]
            top = FROZEN[d].get(q, [])[:10]
            rel = [1.0 if x in gold else 0.0 for x in top]
            pool = set(DENSE[d].get(q, [])[:20]) | set(ce[d].get(q, [])[:20])
            pool -= set(top)
            p = sum(1 for x in pool if x in gold)
            idx = [i for i in range(len(top)) if rel[i] == 0.0 and (slots is None or i in slots)]
            idx.sort(key=lambda i: -W[i])
            pq[q] = base[q] + sum(W[i] for i in idx[:min(p, k)]) / dcg([1.0] * min(len(gold), 10))
    return macro(pq)


union_ladder = {}
for tag, ce in CE_RUNS.items():
    union_ladder[tag] = {
        "k1": ladder_union(ce, 1)[0],
        "k2": ladder_union(ce, 2)[0],
        "k3": ladder_union(ce, 3)[0],
        "k3_slots8_10": ladder_union(ce, 3, slots=set(range(7, 10)))[0],
        "k1_slots8_10": ladder_union(ce, 1, slots=set(range(7, 10)))[0],
    }

# ---- CE-only depth curve (s17 full CE, strongest CE-alone arm available locally)
ce_depth = {}
ce_tag = "s17full" if "s17full" in CE_RUNS else "sft_e2b"
ce = CE_RUNS[ce_tag]
ce_actual = {q: ndcg10(ce[d].get(q, [])[:10], Q[d][q]) for d in DOMAINS for q in QIDS[d]}
CE_ACTUAL, _ = macro(ce_actual)
for depth in (10, 20, 50, 100):
    pq = {}
    for d in DOMAINS:
        for q in QIDS[d]:
            docs = ce[d].get(q, [])[:depth]
            p = sum(1 for x in docs if x in Q[d][q])
            pq[q] = dcg([1.0] * min(p, 10)) / dcg([1.0] * min(len(Q[d][q]), 10))
    ce_depth[depth] = macro(pq)[0]

# same for dense, computed here for the report table
dense_actual = {q: ndcg10(DENSE[d].get(q, [])[:10], Q[d][q]) for d in DOMAINS for q in QIDS[d]}
DENSE_ACTUAL, _ = macro(dense_actual)
dense_depth = {}
for depth in (10, 20, 50, 100, 200, 500, 1000):
    pq = {}
    for d in DOMAINS:
        for q in QIDS[d]:
            docs = DENSE[d].get(q, [])[:depth]
            p = sum(1 for x in docs if x in Q[d][q])
            pq[q] = dcg([1.0] * min(p, 10)) / dcg([1.0] * min(len(Q[d][q]), 10))
    dense_depth[depth] = macro(pq)[0]

# ---- per-domain specialisation: in-sample oracle over measured replacement policies
summary = load("results/replacement_v2/summary_dev.json")
pol = {}
per_pol = {}
for name in ("direct", "contextual", "perdoc", "active"):
    p = summary["policies"].get(name)
    if p:
        pol[name] = p["macro"]
        per_pol[name] = p["per_domain"]
best_policy_per_domain = {}
for d in DOMAINS:
    cands = {name: per_pol[name][d]["policy"] for name in per_pol
             if isinstance(per_pol[name], dict) and isinstance(per_pol[name].get(d), dict)}
    if cands:
        best = max(cands, key=cands.get)
        best_policy_per_domain[d] = {"policy": best, "macro": cands[best],
                                     "baseline": base_per[d],
                                     "delta": cands[best] - base_per[d]}
PER_DOMAIN_POLICY_MACRO = (BASE + sum(v["delta"] for v in best_policy_per_domain.values()) / 13
                           if best_policy_per_domain else None)

out = {
    "baseline_macro": BASE,
    "ordering_only": {
        "macro_if_frozen_top10_perfectly_reordered": REORDER,
        "delta_points": 100 * (REORDER - BASE),
        "n_queries_improvable_by_reordering": n_reorderable,
        "note": "same 10 documents, ideal order; isolates position error from missing-document error",
    },
    "pool_oracle": {
        "macro_ideal_rerank_of_frozen_pool100": POOL,
        "delta_points": 100 * (POOL - BASE),
        "zero_class": dict(zeros),
        "n_zero": nzero,
        "n_zero_with_pool_gold": nzero_pool,
        "zero_class_oracle_macro": ZERO_ORACLE_MACRO,
        "zero_class_oracle_points": 100 * ZERO_ORACLE_POINTS,
        "zero_class_median_points": 100 * ZERO_MEDIAN_POINTS,
        "zero_class_nonzero_mean_points": 100 * ZERO_NZMEAN_POINTS,
        "share_of_pool_headroom_in_zero_class": ZERO_ORACLE_POINTS / (POOL - BASE),
    },
    "union_ladder": {
        "pool": "union(dense top20, CE top20) minus frozen top10; ideal k-slot replacement",
        **union_ladder,
        "committed_rungs": {"depth20_k1_any_slot": 0.5103619641669878,
                            "depth100_k1_dense_only": 0.5832534839716328,
                            "depth100_k2_dense_only": 0.6299273518887838,
                            "depth100_full_rerank": 0.7446226123514618},
    },
    "depth_curves": {
        "dense_actual_k10": DENSE_ACTUAL,
        "dense_oracle_by_depth": dense_depth,
        "ce_actual_k10": CE_ACTUAL,
        "ce_tag": ce_tag,
        "ce_oracle_by_depth": ce_depth,
        "committed_depth_analysis_oracle_column": {"10": 0.3852, "20": 0.4844, "50": 0.6086, "100": 0.7446},
        "note": "committed results/DEPTH_ANALYSIS.md prints recall@k in the oracle column at k=10/20/50; "
                "its k>=100 rows match the official oracle definition",
    },
    "per_domain_policy_specialisation": {
        "policies_measured": pol,
        "best_policy_per_domain": best_policy_per_domain,
        "macro_if_best_policy_per_domain_insample": PER_DOMAIN_POLICY_MACRO,
        "delta_points": (100 * (PER_DOMAIN_POLICY_MACRO - BASE)) if PER_DOMAIN_POLICY_MACRO else None,
    },
}
with open(os.path.join(OUT, "SCORE_DIAGNOSIS_EXTRA.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

print(f"base={BASE:.10f} reorder-only={REORDER:.4f} (+{100*(REORDER-BASE):.2f} pts, {n_reorderable}/519 queries improvable)")
print(f"pool oracle={POOL:.4f} (+{100*(POOL-BASE):.2f}); zero-oracle macro={ZERO_ORACLE_MACRO:.4f} "
      f"(+{100*ZERO_ORACLE_POINTS:.2f} pts = {100*ZERO_ORACLE_POINTS/(POOL-BASE):.1f}% of pool headroom)")
print(f"zeros with pool gold={nzero_pool}/{nzero}; median-lift={100*ZERO_MEDIAN_POINTS:.2f} pts; "
      f"nonzero-mean-lift={100*ZERO_NZMEAN_POINTS:.2f} pts")
print(f"dense actual={DENSE_ACTUAL:.4f} oracle@10..1000=" +
      " ".join(f"{d}:{v:.4f}" for d, v in dense_depth.items()))
print(f"CE({ce_tag}) actual={CE_ACTUAL:.5f} oracle@10..100=" +
      " ".join(f"{d}:{v:.4f}" for d, v in ce_depth.items()))
print("union ladder:", json.dumps(union_ladder))
print(f"per-domain policy oracle={PER_DOMAIN_POLICY_MACRO} delta="
      f"{out['per_domain_policy_specialisation']['delta_points']}")
