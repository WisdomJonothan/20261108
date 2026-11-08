#!/usr/bin/env python3
"""Paired bootstrap over per-query nDCG@10 for two systems on the same topics.

Two resampling units are reported because the benchmark's macro is over topics,
while its domains differ wildly in size (iota 3 dev queries, history 240):

  * query bootstrap  — resample topics with replacement (what a leaderboard CI
    over topics measures);
  * domain-clustered — resample the 13 domains with replacement, then all their
    topics (the honest unit for "would this transfer to another domain set").

Inputs are `score_run.py` JSON files (one per domain per system); topics are
aligned by (domain, qid) and a system that misses a topic gets 0 for it, exactly
like the official scorer.

Usage:
  python bootstrap_compare.py --a "results/score_sub_1_1_*_dev.json" \
      --b "results/ss_dev/score_ss_1_2_*_dev.json" --label "scoresum 1:2 vs RRF 1:1" \
      --json results/boot_scoresum12_vs_rrf11.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(pattern):
    per = {}
    for pat in pattern:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            domain = os.path.basename(d.get("data", "")) or os.path.basename(p)
            for qid, row in d.get("per_query", {}).items():
                per[(domain, qid)] = row["nDCG@10"]
            # topics the run omitted are in the score file only if --allow-missing
            for qid in d.get("topics_missing_from_run", []):
                per[(domain, qid)] = 0.0
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="baseline system score JSONs")
    ap.add_argument("--b", nargs="+", required=True, help="candidate system score JSONs")
    ap.add_argument("--label", default="B vs A")
    ap.add_argument("--metric", default="nDCG@10")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    keys = sorted(set(A) | set(B))
    if not keys:
        raise SystemExit("no per-query rows found")
    a = np.array([A.get(k, 0.0) for k in keys])
    b = np.array([B.get(k, 0.0) for k in keys])
    domains = defaultdict(list)
    for i, (d, _) in enumerate(keys):
        domains[d].append(i)

    rng = np.random.default_rng(args.seed)
    n = len(keys)
    d_query = np.empty(args.iters)
    d_domain = np.empty(args.iters)
    dnames = list(domains)
    for it in range(args.iters):
        idx = rng.integers(0, n, n)
        d_query[it] = b[idx].mean() - a[idx].mean()
        pick = rng.integers(0, len(dnames), len(dnames))
        vals_a, vals_b = [], []
        for j in pick:
            ix = domains[dnames[j]]
            vals_a.append(a[ix].mean())
            vals_b.append(b[ix].mean())
        d_domain[it] = np.mean(vals_b) - np.mean(vals_a)

    def rep(name, base, delta):
        lo, hi = np.percentile(delta, [2.5, 97.5])
        print(f"{name:34s} mean={delta.mean():+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  "
              f"P(Δ>0)={float((delta > 0).mean()):.3f}")
        return {"mean": float(delta.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                "p_gt_0": float((delta > 0).mean())}

    print(f"{args.label}: {len(keys)} topics, {len(domains)} domains")
    print(f"  base topic-macro = {a.mean():.4f}   candidate topic-macro = {b.mean():.4f}   "
          f"observed Δ = {b.mean() - a.mean():+.4f}")
    q = rep("query bootstrap", a.mean(), d_query)
    d = rep("domain-clustered bootstrap", a.mean(), d_domain)
    better = sum(1 for i in range(n) if b[i] > a[i] + 1e-12)
    worse = sum(1 for i in range(n) if b[i] < a[i] - 1e-12)
    print(f"  per-topic: better {better}, worse {worse}, tied {n - better - worse}")

    if args.json:
        out = os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"label": args.label, "topics": n, "domains": len(domains),
                       "baseline_macro": float(a.mean()), "candidate_macro": float(b.mean()),
                       "delta": float(b.mean() - a.mean()), "query_bootstrap": q,
                       "domain_bootstrap": d, "better": better, "worse": worse,
                       "iters": args.iters, "seed": args.seed}, f, indent=1)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
