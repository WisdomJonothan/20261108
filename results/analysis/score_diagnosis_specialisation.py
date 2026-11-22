#!/usr/bin/env python
"""SCORE_DIAGNOSIS part 3: per-domain specialisation ceilings (fusion weight, CE checkpoint),
recomputed from local runs with the frozen z-fusion rule. -> results/analysis/SCORE_DIAGNOSIS_SPECIALISATION.json
"""
import json, math, os
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
    return {q: [(t[2], t[0]) for t in v] for q, v in r.items()}


Q = {d: qrels(d) for d in DOMAINS}
QIDS = {d: sorted(Q[d]) for d in DOMAINS}
DENSE = {d: run(f"runs/dense_mxbai_{d}_dev.trec") for d in DOMAINS}
ARMS = OrderedDict()
for tag, pat in (("sft_e2b", "rerank_sft_e2b_{d}_dev.trec"),
                 ("lx_ep1b", "rerank_lx_ep1b_{d}_dev.trec"),
                 ("sft_e2s29", "rerank_sft_e2s29_{d}_dev.trec"),
                 ("full_s17", "fusion100_full_ce_s17_20261115_{d}_dev_ce.trec"),
                 ("cx_bf", "rerank_cx_bf_{d}_dev.trec")):
    if all(os.path.exists(R("runs/" + pat.format(d=d))) for d in DOMAINS):
        ARMS[tag] = {d: run("runs/" + pat.format(d=d)) for d in DOMAINS}


def zscores(vals):
    mu = sum(vals) / len(vals)
    sd = (sum((x - mu) ** 2 for x in vals) / len(vals)) ** 0.5
    return [(x - mu) / sd if sd > 0 else 0.0 for x in vals]


def fuse(dense_q, ce_q, w_ce):
    """frozen rule: z-normalise both over the intersection, weight 1:w, dense order breaks ties."""
    sd = dict(dense_q)
    sc = dict(ce_q)
    common = [d for d, _ in dense_q if d in sc]
    za = zscores([sd[d] for d in common])
    zb = zscores([sc[d] for d in common])
    book = {d: a + w_ce * b for d, a, b in zip(common, za, zb)}
    return sorted(common, key=lambda d: -book[d])


WEIGHTS = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.0]
out = {"weights": WEIGHTS, "arms": list(ARMS), "per_arm_weight_sweep": {}, "per_domain": {}}

for tag, ce in ARMS.items():
    per_w = {}
    for w in WEIGHTS:
        pq = {}
        for d in DOMAINS:
            for q in QIDS[d]:
                pq[q] = ndcg10(fuse(DENSE[d].get(q, []), ce[d].get(q, []), w), Q[d][q])
        per_dom = {d: sum(pq[q] for q in QIDS[d]) / len(QIDS[d]) for d in DOMAINS}
        per_w[w] = {"macro": sum(per_dom.values()) / 13, "per_domain": per_dom}
    out["per_arm_weight_sweep"][tag] = per_w

# per-arm, w=1.0 per-domain table + oracle over w per domain
w1 = {tag: out["per_arm_weight_sweep"][tag][1.0] for tag in ARMS}
out["at_w1"] = {tag: w1[tag]["macro"] for tag in ARMS}
best_arm_w1 = max(out["at_w1"], key=out["at_w1"].get)
out["best_arm_at_w1"] = {"arm": best_arm_w1, "macro": out["at_w1"][best_arm_w1]}

# global best weight over arms (in-sample)
glob = [(tag, w, out["per_arm_weight_sweep"][tag][w]["macro"]) for tag in ARMS for w in WEIGHTS]
tag_g, w_g, m_g = max(glob, key=lambda t: t[2])
out["global_best"] = {"arm": tag_g, "w_ce": w_g, "macro": m_g}

# per-domain oracle over (arm, w) -- the per-domain specialisation ceiling
dom_pick = {}
for d in DOMAINS:
    cands = {(tag, w): out["per_arm_weight_sweep"][tag][w]["per_domain"][d] for tag in ARMS for w in WEIGHTS}
    key = max(cands, key=cands.get)
    dom_pick[d] = {"arm": key[0], "w_ce": key[1], "macro": cands[key]}
out["per_domain_oracle_over_arm_and_weight"] = {
    "picks": dom_pick,
    "macro": sum(v["macro"] for v in dom_pick.values()) / 13,
    "n_distinct_arms": len({v["arm"] for v in dom_pick.values()}),
    "n_distinct_weights": len({v["w_ce"] for v in dom_pick.values()}),
}
# per-domain oracle over weight only, holding the strongest arm at w=1
w_only = {}
for d in DOMAINS:
    cands = {w: out["per_arm_weight_sweep"][best_arm_w1][w]["per_domain"][d] for w in WEIGHTS}
    wb = max(cands, key=cands.get)
    w_only[d] = {"w_ce": wb, "macro": cands[wb]}
out["per_domain_oracle_over_weight_only"] = {"picks": w_only,
                                             "macro": sum(v["macro"] for v in w_only.values()) / 13}
# per-domain oracle over arm only, holding w=1
a_only = {}
for d in DOMAINS:
    cands = {tag: out["per_arm_weight_sweep"][tag][1.0]["per_domain"][d] for tag in ARMS}
    ab = max(cands, key=cands.get)
    a_only[d] = {"arm": ab, "macro": cands[ab]}
out["per_domain_oracle_over_arm_only"] = {"picks": a_only,
                                          "macro": sum(v["macro"] for v in a_only.values()) / 13}

with open(os.path.join(OUT, "SCORE_DIAGNOSIS_SPECIALISATION.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

print("arms:", list(ARMS))
print("at w=1:", {k: round(v, 4) for k, v in out["at_w1"].items()})
print(f"global best: {tag_g} w={w_g} macro={m_g:.4f}")
print(f"baseline (frozen e2bc, w=1) 0.3580985233; per-domain oracle over (arm,w) = "
      f"{out['per_domain_oracle_over_arm_and_weight']['macro']:.4f} "
      f"({out['per_domain_oracle_over_arm_and_weight']['n_distinct_arms']} arms, "
      f"{out['per_domain_oracle_over_arm_and_weight']['n_distinct_weights']} weights)")
print(f"per-domain oracle weight only (arm {best_arm_w1}) = {out['per_domain_oracle_over_weight_only']['macro']:.4f}")
print(f"per-domain oracle arm only (w=1) = {out['per_domain_oracle_over_arm_only']['macro']:.4f}")
for d in DOMAINS:
    p = out["per_domain_oracle_over_arm_and_weight"]["picks"][d]
    print(f"  {d:10s} base={out['per_arm_weight_sweep']['sft_e2b'][1.0]['per_domain'][d]:.4f} "
          f"pick={p['arm']}@{p['w_ce']} {p['macro']:.4f}")
