#!/usr/bin/env python3
"""CPU replay of the committed replacement results (no GPU, no model).

Rebuilds every reported number from the artefacts in results/replacement_v2/
plus the official dev qrels in data/track1_tempo/<domain>/qrels_dev.txt:

  * per-policy nDCG@10, domain-macro, paired-bootstrap CI and swap counts
    (from pair_details + pair_records, using the frozen top-10 in the
    candidates file),
  * the frozen-pool oracle (best single swap per query),
  * the decision-rule ablations (full / no_abstain / no_bidir / neither)
    from exp_direct_probs_dev.json, which stores both orders' A/B/C
    probabilities per pair.

Every recomputed value is compared with the committed one; the script exits
non-zero if any disagrees beyond 1e-6, so a clean checkout can verify the
reported table on a CPU.

Usage:
  python scripts/replay_replacement.py            # verify everything
  python scripts/replay_replacement.py --quick    # skip the 10k bootstrap
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/replacement_v2"
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]
DRAWS, SEED = 10000, 20261120


def read_qrels(domain):
    gold = {}
    with (ROOT / "data/track1_tempo" / domain / "qrels_dev.txt").open(encoding="utf-8") as fh:
        for ln in fh:
            p = ln.split()
            if len(p) >= 4 and int(p[3]) > 0:
                gold.setdefault(p[0], set()).add(p[2])
    return gold


def ndcg10(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def macro(per_query, key):
    return float(np.mean([np.mean([r[key] for r in per_query if r["domain"] == d])
                          for d in DOMAINS]))


def bootstrap(per_query, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    deltas = {d: np.array([r["delta"] for r in per_query if r["domain"] == d]) for d in DOMAINS}
    point = float(np.mean([v.mean() for v in deltas.values()]))
    sims = np.empty(draws)
    for i in range(draws):
        sims[i] = np.mean([v[rng.integers(len(v), size=len(v))].mean() for v in deltas.values()])
    return point, [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))]


def load(name):
    p = V2 / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def candidates_for(exp):
    """Candidate file an experiment was run on (falls back to the frozen one)."""
    name = Path(exp.get("candidate_file", "candidates_dev.json")).name
    return name if (V2 / name).exists() else "candidates_dev.json"


def load_candidates(name):
    """Key by (domain, qid): FollowIR's two instruction conditions share query ids."""
    rows = load(name)["queries"]
    out = {}
    for row in rows:
        row["gold"] = read_qrels(row["domain"]).get(row["qid"], set())
        key = (row["domain"], str(row["qid"]))
        assert key not in out, f"duplicate (domain, qid): {key}"
        out[key] = row
    return out


def replay_policy(exp, policy, cand_by_key):
    pol = exp["policies"][policy]
    details = {}
    for rec in exp["pair_details"]:
        for p in rec["pairs"]:
            details[(rec["domain"], str(rec["qid"]), p["slot_rank"], round(p["delta_ndcg"], 9))] = p
    chosen = {}
    for r in pol["pair_records"]:
        if not r.get("chosen"):
            continue
        key = (r["domain"], str(r["qid"]), r["slot_rank"], round(r["delta_ndcg"], 9))
        p = details[key]
        chosen[(r["domain"], str(r["qid"]))] = (p["slot_rank"] - 1, p["challenger"])
    per_query, counts = [], {"helpful": 0, "harmful": 0, "neutral": 0, "no_swap": 0}
    gained = lost = 0.0
    for key, row in cand_by_key.items():
        top = list(row["frozen_top10"])
        base = ndcg10(top, row["gold"])
        ranking = list(top)
        if key in chosen:
            slot, doc = chosen[key]
            ranking[slot] = doc
        nd = ndcg10(ranking, row["gold"])
        delta = nd - base
        if key in chosen:
            if delta > 1e-12:
                counts["helpful"] += 1
            elif delta < -1e-12:
                counts["harmful"] += 1
            else:
                counts["neutral"] += 1
        gained += max(delta, 0.0)
        lost += min(delta, 0.0)
        per_query.append({"domain": row["domain"], "qid": row["qid"], "baseline": base,
                          "policy": nd, "delta": delta})
    counts["no_swap"] = len(per_query) - len(chosen)
    return per_query, counts, gained, lost


def replay_oracle(cand_by_key):
    per_domain = {d: [] for d in DOMAINS}
    positive = 0
    pairs_pos = pairs_neg = 0
    for qid, row in cand_by_key.items():
        top = row["frozen_top10"]
        base = ndcg10(top, row["gold"])
        qbest = 0.0
        for p in row["pairs"]:
            ranking = list(top)
            ranking[p["slot_rank"] - 1] = p["incoming_docid"] if "incoming_docid" in p else p["challenger"]
            d = ndcg10(ranking, row["gold"]) - base
            if d > 1e-12:
                pairs_pos += 1
            elif d < -1e-12:
                pairs_neg += 1
            qbest = max(qbest, d)
        if qbest > 0:
            positive += 1
        per_domain[row["domain"]].append(qbest)
    return float(np.mean([np.mean(v) for v in per_domain.values() if v])), positive, pairs_pos, pairs_neg


def norm_ab(p):
    s = p["incumbent"] + p["challenger"]
    if s <= 0:
        return {"incumbent": 0.5, "challenger": 0.5}
    return {"incumbent": p["incumbent"] / s, "challenger": p["challenger"] / s}


def choose_pair(pairs, rule, threshold=0.0):
    best = None
    for p in pairs:
        pred = p.get("pred") or {}
        fwd, rev = pred.get("fwd"), pred.get("rev")
        if not fwd:
            continue
        if rule in ("no_abstain", "neither"):
            fwd = norm_ab(fwd)
            rev = norm_ab(rev) if rev else None
        orders = [fwd] if rule in ("no_bidir", "neither") else ([fwd, rev] if rev else [fwd])
        merged = sum(o["challenger"] for o in orders) / len(orders)
        agree = all(max(o, key=o.get) == "challenger" for o in orders)
        if agree and merged - 0.5 > threshold and (best is None or merged > best[0]):
            best = (merged, p)
    return best


def replay_ablation(probs, rule, cand_by_key):
    per_query = []
    counts = {"helpful": 0, "harmful": 0, "neutral": 0, "no_swap": 0}
    for rec in probs["pair_details"]:
        row = cand_by_key[(rec["domain"], str(rec["qid"]))]
        got = choose_pair(rec["pairs"], rule)
        if got is None:
            delta = 0.0
            counts["no_swap"] += 1
        else:
            delta = float(got[1]["delta_ndcg"])
            counts["helpful" if delta > 1e-12 else "harmful" if delta < -1e-12 else "neutral"] += 1
        per_query.append({"domain": row["domain"], "qid": rec["qid"],
                          "baseline": rec.get("baseline", 0.0),
                          "policy": rec.get("baseline", 0.0) + delta,
                          "delta": delta})
    return per_query, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip the 10k bootstrap")
    args = ap.parse_args()

    cand_by_key = load_candidates("candidates_dev.json")
    print("candidates: %d queries, %d pairs" %
          (len(cand_by_key), sum(len(r["pairs"]) for r in cand_by_key.values())))

    failures = []

    def check(label, got, want, tol=1e-6):
        ok = abs(got - want) <= tol
        print("%-46s %+.6f  (committed %+.6f) %s" % (label, got, want, "OK" if ok else "MISMATCH"))
        if not ok:
            failures.append(label)

    for name, policies in (("exp_direct_context_dev.json", ["direct", "contextual"]),
                           ("exp_perdoc_dev.json", ["perdoc"]),
                           ("exp_active_dev.json", ["active"]),
                           ("exp_budget_cap1.json", ["direct"]),
                           ("exp_budget_cap2.json", ["direct"]),
                           ("exp_dense_start_dev.json", ["direct"]),
                           ("exp_rr_start_dev.json", ["direct"])):
        exp = load(name)
        if exp is None:
            print("skip", name)
            continue
        for pol in policies:
            if pol not in exp["policies"]:
                continue
            cname = candidates_for(exp)
            cur = cand_by_key if cname == "candidates_dev.json" else load_candidates(cname)
            per_query, counts, gained, lost = replay_policy(exp, pol, cur)
            committed = exp["policies"][pol]
            tag = "%s/%s" % (name.replace("exp_", "").replace("_dev.json", ""), pol)
            if cname != "candidates_dev.json":
                tag += " [" + cname.replace("candidates_", "").replace(".json", "") + "]"
            check(tag + " macro", macro(per_query, "policy"), committed["macro_policy"])
            for k, v in counts.items():
                if v != committed["swap_counts"].get(k):
                    failures.append("%s count %s (%d vs %d)" % (tag, k, v,
                                                                committed["swap_counts"].get(k)))
            if abs(gained - committed["points_gained"]) > 1e-6:
                failures.append(tag + " gained")
            if not args.quick:
                point, ci = bootstrap(per_query)
                b = committed["bootstrap"]
                check(tag + " delta", point, b["delta"])
                print("    replay CI [%+.6f, %+.6f]  committed [%+.6f, %+.6f]"
                      % (ci[0], ci[1], b["ci95"][0], b["ci95"][1]))

    oracle = load("oracle_fixed_dev.json")
    if oracle:
        oc = oracle.get("candidate_file")
        okey = cand_by_key if not oc or Path(oc).name == "candidates_dev.json" else load_candidates(Path(oc).name)
        gain, positive, pos_pairs, neg_pairs = replay_oracle(okey)
        check("oracle gain", gain, oracle["macro_gain"])
        print("    queries with a beneficial pair: %d (committed %d); pairs +%d/-%d"
              % (positive, oracle["queries_with_beneficial_pair"], pos_pairs, neg_pairs))

    probs = load("exp_direct_probs_dev.json")
    abl = load("ablations_dev.json")
    if probs and abl:
        for rule in ("full", "no_abstain", "no_bidir", "neither"):
            per_query, counts = replay_ablation(probs, rule, cand_by_key)
            check("ablation %s macro" % rule, macro(per_query, "policy"),
                  abl["rules"][rule]["macro_policy"])
            committed = abl["rules"][rule]["swap_counts"]
            for k in ("helpful", "harmful", "neutral", "no_swap"):
                if counts[k] != committed.get(k):
                    failures.append("ablation %s %s (%d vs %d)" %
                                    (rule, k, counts[k], committed.get(k)))

    if failures:
        print("\nFAIL:", ", ".join(failures[:12]))
        sys.exit(1)
    print("\nRESULT: replay matches every committed number")


if __name__ == "__main__":
    main()
