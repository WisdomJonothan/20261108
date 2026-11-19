#!/usr/bin/env python3
"""Paired bootstrap between two policies stored in one exp_*.json.

Per-query nDCG@10 change of a policy = sum of delta_ndcg over chosen swap
records (one swap per query, so at most one non-zero term).
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

DOMAINS = ["bitcoin","cardano","economics","genealogy","history","hsm","iota",
           "law","monero","politics","quant","travel","workplace"]

def policy_deltas(pol):
    d = {}
    for r in pol["pair_records"]:
        key = (r["domain"], r["qid"])
        d.setdefault(key, 0.0)
        if r.get("chosen"):
            d[key] += r["delta_ndcg"]
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261120)
    a = ap.parse_args()
    blob = json.loads(Path(a.file).read_text())
    pa, pb = blob["policies"][a.a], blob["policies"][a.b]
    da, db = policy_deltas(pa), policy_deltas(pb)
    keys = sorted(set(da) | set(db))
    per_domain = defaultdict(list)
    for k in keys:
        per_domain[k[0]].append(db.get(k,0.0) - da.get(k,0.0))
    doms = [d for d in DOMAINS if per_domain[d]]
    base = float(np.mean([np.mean(per_domain[d]) for d in doms]))
    rng = np.random.default_rng(a.seed)
    sims = np.empty(a.draws)
    for i in range(a.draws):
        sims[i] = np.mean([np.mean(np.array(per_domain[d])[rng.integers(len(per_domain[d]), size=len(per_domain[d]))]) for d in doms])
    lo, hi = np.percentile(sims, [2.5, 97.5])
    win = sum(1 for k in keys if db.get(k,0.0) - da.get(k,0.0) > 1e-9)
    loss = sum(1 for k in keys if db.get(k,0.0) - da.get(k,0.0) < -1e-9)
    print(json.dumps({"a": a.a, "b": a.b, "delta_b_minus_a": round(base,6),
                      "ci95": [round(float(lo),6), round(float(hi),6)],
                      "p_le_0": round(float(np.mean(sims <= 0)),4),
                      "queries_where_b_better": win, "queries_where_a_better": loss,
                      "n_queries": len(keys)}, indent=1))

if __name__ == "__main__":
    main()
