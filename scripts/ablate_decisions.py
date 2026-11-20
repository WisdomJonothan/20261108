"""Decision-rule ablations computed post-hoc from saved per-order judge probabilities.

Input: an exp json produced by benchmark_pairwise_probs.py (pair_details[*].pairs[*].pred
has fwd/rev dicts with incumbent / challenger / abstain probabilities).

Rules
  full            bidirectional + abstain (the frozen rule; must reproduce the run)
  no_abstain      abstain option removed, A/B probabilities renormalised
  no_bidir        forward order only, abstain kept
  neither         forward order only, no abstain
Each rule may perform at most one swap per query (the highest merged challenger
probability among eligible pairs, threshold 0).

Output: results/replacement_v2/ablations_dev.json with macro nDCG@10, delta,
paired bootstrap CI (10k draws, seed 20261120, queries resampled within domain)
and swap counts for every rule.
"""
import argparse
import json
import random
from pathlib import Path

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]


def norm_ab(p):
    s = p["incumbent"] + p["challenger"]
    if s <= 0:
        return {"incumbent": 0.5, "challenger": 0.5}
    return {"incumbent": p["incumbent"] / s, "challenger": p["challenger"] / s}


def choose(pairs, rule, threshold=0.0):
    best = None
    for p in pairs:
        pred = p.get("pred") or {}
        fwd, rev = pred.get("fwd"), pred.get("rev")
        if not fwd:
            continue
        if rule in ("no_abstain", "neither"):
            fwd = norm_ab(fwd)
            rev = norm_ab(rev) if rev else None
        if rule in ("no_bidir", "neither"):
            orders = [fwd]
        else:
            orders = [fwd, rev] if rev else [fwd]
        merged = sum(o["challenger"] for o in orders) / len(orders)
        agree = all(max(o, key=o.get) == "challenger" for o in orders)
        if agree and merged - 0.5 > threshold and (best is None or merged > best[0]):
            best = (merged, p)
    return best


def macro(rows):
    per_domain = {}
    for d in DOMAINS:
        vals = [r["policy"] for r in rows if r["domain"] == d]
        if vals:
            per_domain[d] = sum(vals) / len(vals)
    return sum(per_domain.values()) / len(per_domain), per_domain


def boot(rows, rng, n=10000):
    by = {d: [r["delta"] for r in rows if r["domain"] == d] for d in DOMAINS}
    draws = []
    for _ in range(n):
        tot = 0.0
        for d in DOMAINS:
            v = by[d]
            if not v:
                continue
            tot += sum(rng.choice(v) for _ in v) / len(v)
        draws.append(tot / len(DOMAINS))
    draws.sort()
    return [draws[int(0.025 * n)], draws[int(0.975 * n)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--exp", default="results/replacement_v2/exp_direct_probs_dev.json")
    ap.add_argument("--out", default="results/replacement_v2/ablations_dev.json")
    args = ap.parse_args()
    root = args.root
    data = json.loads((root / args.exp).read_text())
    details = data["pair_details"]
    rng = random.Random(20261120)
    out = {"exp": args.exp, "n_queries": len(details), "rules": {}}
    for rule in ("full", "no_abstain", "no_bidir", "neither"):
        rows, counts = [], {"helpful": 0, "harmful": 0, "neutral": 0, "no_swap": 0}
        for q in details:
            got = choose(q["pairs"], rule)
            if got is None:
                delta, swapped = 0.0, False
                counts["no_swap"] += 1
            else:
                _, pair = got
                delta, swapped = float(pair["delta_ndcg"]), True
                if delta > 1e-12:
                    counts["helpful"] += 1
                elif delta < -1e-12:
                    counts["harmful"] += 1
                else:
                    counts["neutral"] += 1
            rows.append({"domain": q["domain"], "qid": q["qid"],
                         "baseline": q["baseline"], "delta": delta,
                         "policy": q["baseline"] + delta, "swap": swapped})
        m_pol, per_domain = macro(rows)
        m_base, _ = macro([{**r, "policy": r["baseline"]} for r in rows])
        out["rules"][rule] = {
            "macro_policy": m_pol,
            "macro_baseline": m_base,
            "macro_delta": m_pol - m_base,
            "ci95": boot(rows, rng),
            "swap_counts": counts,
            "points_gained": sum(r["delta"] for r in rows if r["delta"] > 0),
            "points_lost": sum(r["delta"] for r in rows if r["delta"] < 0),
            "per_domain": per_domain,
        }
    (root / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for rule, r in out["rules"].items():
        print(f"{rule:12s} macro {r['macro_policy']:.6f}  delta {r['macro_delta']:+.6f} "
              f"ci95 [{r['ci95'][0]:+.5f},{r['ci95'][1]:+.5f}]  {r['swap_counts']}")
    print("wrote", root / args.out)


if __name__ == "__main__":
    main()
