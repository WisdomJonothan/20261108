"""Where the six-pair configuration loses: ceiling ladder + rejected-beneficial dump.

Replays the frozen decision rule over the cached 18 judged actions per dev query
(results/replacement_v2/expanded3_fixed_dev.json) and reports, all under the same
one-swap, no-op-allowed protocol:

  base                  frozen fused system (0.358099)
  current               head-first six pairs, rule = agree & mp>0.5, best mp wins
  select_accepted       same accept set, but pick the *best* accepted action by true delta
  perfect_six           perfect judgment + selection inside the six judged pairs
  perfect_eighteen      perfect judgment + selection inside all eighteen cached pairs

Also dumps every query where a beneficial action exists inside the six pairs but the
rule accepts none of them (the "44" class the reviewer identified), with the action
context needed for attribution.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

DOMAINS = None


def dom_macro(per_query):
    by = defaultdict(list)
    for r in per_query:
        by[r["domain"]].append(r["value"])
    doms = sorted(by)
    means = [sum(v) / len(v) for v in by.values()]
    return sum(means) / len(means), doms


def order_actions(acts, mode):
    if mode == "headfirst":
        return sorted(acts, key=lambda a: (a["slot"], -a["prior"]))
    if mode == "prior":
        return sorted(acts, key=lambda a: -a["prior"])
    raise SystemExit(f"unknown order {mode}")


def choose(acts, threshold=0.0):
    """Frozen rule: both orders agree on the challenger, margin>threshold, best margin wins."""
    elig = [a for a in acts if a["agree"] and (a["mp"] - 0.5) > threshold]
    if not elig:
        return None
    return max(elig, key=lambda a: a["mp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/replacement_v2/expanded3_fixed_dev.json")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="results/replacement_v2/oracle_ladder_dev.json")
    ap.add_argument("--dump", default="results/replacement_v2/rejected_beneficial_dev.json")
    a = ap.parse_args()

    d = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    pool = d["budgets"][str(6)]["per_query"] if "6" in d["budgets"] else None
    rows = d["budgets"][str(a.k)]["per_query"] if str(a.k) in d["budgets"] else pool
    if rows is None:
        raise SystemExit("cache has no per_query rows")

    cur, sel, per6, per18, rejected = [], [], [], [], []
    for r in rows:
        base = r["baseline"]
        acts = r["actions"]
        first = order_actions(acts, "headfirst")[: a.k]
        pick = choose(first)
        cur.append({"domain": r["domain"], "qid": r["qid"], "value": base + (pick["delta"] if pick else 0.0)})

        acc = [x for x in acts[:0]]  # placeholder for clarity
        acc = [x for x in first if x["agree"] and x["mp"] - 0.5 > 0.0]
        if acc:
            best = max(acc, key=lambda x: x["delta"])
            sel.append({"domain": r["domain"], "qid": r["qid"], "value": base + max(0.0, best["delta"])})
        else:
            sel.append({"domain": r["domain"], "qid": r["qid"], "value": base})

        b6 = max([x["delta"] for x in first] + [0.0])
        b18 = max([x["delta"] for x in acts] + [0.0])
        per6.append({"domain": r["domain"], "qid": r["qid"], "value": base + b6})
        per18.append({"domain": r["domain"], "qid": r["qid"], "value": base + b18})

        beneficial = [x for x in first if x["delta"] > 1e-12]
        if beneficial and not acc:
            rejected.append({
                "domain": r["domain"], "qid": r["qid"], "baseline": base,
                "best_available_delta": max(x["delta"] for x in beneficial),
                "macro_weight": 1.0 / 13.0,
                "actions": [{k: x[k] for k in ("slot", "incumbent", "challenger", "prior", "mp", "agree", "p_fwd", "p_rev", "delta")} for x in first],
            })

    base_macro, doms = dom_macro([{"domain": r["domain"], "value": rows[i]["baseline"]} for i, r in enumerate(rows)])
    out = {"k": a.k, "order": "headfirst", "domains": doms, "baseline": base_macro}
    for name, rs in (("current", cur), ("select_accepted", sel), ("perfect_six", per6), ("perfect_eighteen", per18)):
        m, _ = dom_macro(rs)
        out[name] = round(m, 6)
        out[name + "_gain"] = round(m - base_macro, 6)
    out["queries"] = len(rows)
    out["queries_with_beneficial_in_pairs"] = sum(1 for r in rows if any(x["delta"] > 1e-12 for x in order_actions(r["actions"], "headfirst")[: a.k]))
    out["queries_rejecting_all_beneficial"] = len(rejected)
    out["rejected_best_delta_sum"] = round(sum(x["best_available_delta"] for x in rejected), 4)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    Path(a.dump).write_text(json.dumps(rejected, indent=1), encoding="utf-8")

    print(f"k={a.k} order=headfirst n={out['queries']}")
    print(f"  base                 {base_macro:.6f}")
    for name in ("current", "select_accepted", "perfect_six", "perfect_eighteen"):
        print(f"  {name:<20} {out[name]:.6f}  (+{out[name+'_gain']:.6f})")
    print(f"  beneficial in pairs {out['queries_with_beneficial_in_pairs']}  rejected-all {out['queries_rejecting_all_beneficial']}")
    print("RESULT: ladder_written" if Path(a.out).exists() else "RESULT: failed")


if __name__ == "__main__":
    main()
