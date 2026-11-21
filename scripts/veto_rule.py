"""Veto rule: keep the relevance judge's chosen swap unless the condition question
clearly prefers the incumbent.

Replays the two cached judgement sets. For every query the Direct arm's chosen pair is
kept when the condition judge's merged challenger probability exceeds a threshold (or
is at least not confidently on the incumbent side); otherwise the query keeps its
original ranking (no swap). Sweeps the veto threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ensemble_rule as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:/RETECO/github_repo")
    ap.add_argument("--direct", default="results/replacement_v2/exp_direct_probs_dev.json")
    ap.add_argument("--condition", default="results/replacement_v2/exp_condition_pairs_dev.json")
    ap.add_argument("--out", default="results/replacement_v2/veto_rules_dev.json")
    args = ap.parse_args()

    root = Path(args.root)
    direct = E.load_direct(root / args.direct)
    cond = E.load_condition(root / args.condition)

    rows = []
    for key, det in direct.items():
        cond_pred = {(p["slot_rank"], p["challenger"]): p["pred"] for p in cond.get(key, [])}
        chosen = None
        for p in det["pairs"]:
            if p.get("agree") and p.get("merged_challenger", 0.0) - 0.5 > 0.0:
                if chosen is None or p["merged_challenger"] > chosen["merged_challenger"]:
                    chosen = p
        entry = {"domain": key[0], "qid": key[1], "baseline": det["baseline"], "delta": 0.0}
        if chosen is not None:
            cp = cond_pred.get((chosen["slot_rank"], chosen["challenger"]))
            entry["delta"] = chosen["delta_ndcg"]
            entry["cond_merged"] = E.merged(cp) if cp else None
            entry["cond_incumbent"] = bool(cp) and not E.eligible(cp)
        rows.append(entry)

    out = {"rules": {}}
    grid = [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    for tau in grid:
        per_query = []
        stats = {"swaps": 0, "helpful": 0, "harmful": 0, "neutral": 0}
        for r in rows:
            delta = 0.0
            if "cond_merged" in r:
                cm = r["cond_merged"]
                keep = (cm is None) or (cm >= tau) or (not r.get("cond_incumbent"))
                delta = r["delta"] if keep else 0.0
            policy = r["baseline"] + delta if delta else r["baseline"]
            if delta > 1e-9:
                stats["helpful"] += 1
                stats["swaps"] += 1
            elif delta < -1e-9:
                stats["harmful"] += 1
                stats["swaps"] += 1
            elif abs(r.get("delta", 0.0)) > 0:
                stats["neutral"] += 1
                stats["swaps"] += 1
            per_query.append({"domain": r["domain"], "qid": r["qid"], "baseline": r["baseline"], "policy": policy})
        macro = E.domain_macro(per_query)
        base = E.domain_macro([{**p, "policy": p["baseline"]} for p in per_query])
        boot = E.paired_bootstrap(per_query)
        out["rules"][str(tau)] = {
            "macro_policy": macro,
            "macro_delta": macro - base,
            "bootstrap": boot,
            "stats": stats,
        }
    dest = root / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for tau, v in out["rules"].items():
        print(f"veto tau={tau} delta={v['macro_delta']:+.6f} ci={[round(x, 5) for x in v['bootstrap']['ci95']]} "
              f"swaps={v['stats']['swaps']} H={v['stats']['helpful']} X={v['stats']['harmful']}")


if __name__ == "__main__":
    main()
