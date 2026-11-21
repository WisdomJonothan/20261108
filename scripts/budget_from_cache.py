#!/usr/bin/env python3
"""Budget curve replayed from a single prediction cache (no GPU, no model).

The Direct run already judged every one of the 4,671 pairs in both orders, so
we can ask what a budget-constrained policy would have done without paying for
new inference: give it only the first k = 1, 2, 3, 6, 9 pairs of a query under
some ordering, let it apply the frozen decision rule to those, and measure the
result. Because every arm reads the same stored probabilities, differences
between arms are differences in pair selection, not sampling noise.

Orderings compared
  fixed     slot order (rank 8, 9, 10) as the candidate file lists them
  cheapgap  ascending |dense score delta + cross-encoder z delta| (the frozen
            active-control ordering)
  random    a seeded random permutation
  uncertain ascending |merged challenger probability - 0.5|, the AcuRank-style
            uncertainty ordering adapted to our judge
  oracle    pairs sorted by their true nDCG delta (upper bound for that k)

Outcome: results/replacement_v2/budget_cache_dev.json and a markdown table on
stdout: for each k and ordering, domain-macro nDCG@10, delta vs frozen, a
paired bootstrap CI over 10k draws (seed 20261120, queries resampled within
domain) and the judge calls actually spent.

Usage: python scripts/budget_from_cache.py [--k 1 2 3 6 9]
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/replacement_v2"
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]
SEED = 20261120


def ndcg10(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def read_gold(domain):
    gold = {}
    with (ROOT / "data/track1_tempo" / domain / "qrels_dev.txt").open(encoding="utf-8") as fh:
        for ln in fh:
            p = ln.split()
            if len(p) >= 4 and int(p[3]) > 0:
                gold.setdefault(p[0], set()).add(p[2])
    return gold


def cheap_gap(pair):
    d = (pair.get("incoming_dense_score") or 0.0) - (pair.get("removed_dense_score") or 0.0)
    c = (pair.get("incoming_ce_z") or 0.0) - (pair.get("removed_ce_z") or 0.0)
    return d + c


def merged_and_agree(pred):
    fwd, rev = pred.get("fwd"), pred.get("rev")
    merged = 0.5 * (fwd["challenger"] + rev["challenger"])
    agree = (max(fwd, key=fwd.get) == "challenger" and max(rev, key=rev.get) == "challenger")
    return merged, agree


def macro(rows):
    per = {d: [r["policy"] for r in rows if r["domain"] == d] for d in DOMAINS}
    return float(np.mean([np.mean(v) for v in per.values() if v]))


def boot(rows, draws=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    deltas = {d: np.array([r["delta"] for r in rows if r["domain"] == d]) for d in DOMAINS}
    sims = np.empty(draws)
    for i in range(draws):
        sims[i] = np.mean([v[rng.integers(len(v), size=len(v))].mean()
                           for v in deltas.values() if len(v)])
    return [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 6, 9])
    ap.add_argument("--out", default="results/replacement_v2/budget_cache_dev.json")
    args = ap.parse_args()

    cand = json.loads((V2 / "candidates_dev.json").read_text(encoding="utf-8"))
    probs = json.loads((V2 / "exp_direct_probs_dev.json").read_text(encoding="utf-8"))
    gold_by_domain = {d: read_gold(d) for d in DOMAINS}
    pred_by_key = {}
    for rec in probs["pair_details"]:
        for p in rec["pairs"]:
            pred_by_key[(rec["qid"], p["slot_rank"], p["challenger"])] = p

    rows = []
    for q in cand["queries"]:
        gold = gold_by_domain[q["domain"]].get(q["qid"], set())
        top = q["frozen_top10"]
        base = ndcg10(top, gold)
        pairs = []
        for p in q["pairs"]:
            pr = pred_by_key.get((q["qid"], p["slot_rank"], p["incoming_docid"]))
            if pr is None:
                continue
            merged, agree = merged_and_agree(pr["pred"])
            pairs.append({"slot_rank": p["slot_rank"], "challenger": p["incoming_docid"],
                          "gap": abs(cheap_gap(p)), "merged": merged, "agree": agree,
                          "delta": pr["delta_ndcg"]})
        rng = random.Random(SEED + hash(q["qid"]) % 100000)
        shuff = list(range(len(pairs)))
        rng.shuffle(shuff)
        rows.append({"domain": q["domain"], "qid": q["qid"], "gold": gold, "top": top,
                     "base": base, "pairs": pairs,
                     "order": {"fixed": list(range(len(pairs))),
                               "cheapgap": sorted(range(len(pairs)), key=lambda i: pairs[i]["gap"]),
                               "random": shuff,
                               "uncertain": sorted(range(len(pairs)),
                                                   key=lambda i: abs(pairs[i]["merged"] - 0.5)),
                               "oracle": sorted(range(len(pairs)), key=lambda i: -pairs[i]["delta"])}})

    results = {}
    for name in ("fixed", "cheapgap", "uncertain", "random", "oracle"):
        for k in args.k:
            per_query, calls = [], 0
            for r in rows:
                ranking = list(r["top"])
                best = None
                for idx in r["order"][name][:k]:
                    p = r["pairs"][idx]
                    calls += 1
                    if not (p["agree"] and p["merged"] > 0.5):
                        continue
                    key = (p["merged"], -p["slot_rank"], p["challenger"])
                    if best is None or key > best[0]:
                        best = (key, p)
                swapped = best is not None
                if swapped:
                    ranking[best[1]["slot_rank"] - 1] = best[1]["challenger"]
                nd = ndcg10(ranking, r["gold"])
                per_query.append({"domain": r["domain"], "qid": r["qid"],
                                  "baseline": r["base"], "policy": nd,
                                  "delta": nd - r["base"], "swap": swapped})
            m = macro(per_query)
            base_macro = float(np.mean([np.mean([r["baseline"] for r in per_query
                                                 if r["domain"] == d]) for d in DOMAINS]))
            d = m - base_macro
            entry = {"k": k, "macro": m, "delta": d, "ci95": boot(per_query),
                     "judge_calls": calls, "prompts": 2 * calls,
                     "swaps": sum(1 for r in per_query if r["swap"])}
            results.setdefault(name, []).append(entry)
            print("%-9s k=%d  macro %.6f  delta %+.6f  CI [%+.6f, %+.6f]  calls %d"
                  % (name, k, m, d, entry["ci95"][0], entry["ci95"][1], calls))

    frozen = float(np.mean([np.mean([r["base"] for r in rows if r["domain"] == d])
                            for d in DOMAINS]))
    payload = {"candidate_file": "candidates_dev.json",
               "prediction_cache": "exp_direct_probs_dev.json",
               "frozen_macro": frozen, "seed": SEED, "curves": results}
    (ROOT / args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
