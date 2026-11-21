"""Accept a swap only when BOTH questions independently prefer the challenger.

Replays two cached judgement sets (relevance question from the Direct arm, condition
question from the condition arm) over the same 4,671 pairs, and applies the frozen
rule (challenger argmax in both presentation orders, merged margin > 0.5, best merged
wins, at most one swap per query).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DOMAINS = [
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
    "law", "monero", "politics", "quant", "travel", "workplace",
]


def merged(pred):
    o = {}
    for order in ("fwd", "rev"):
        p = pred[order]
        tot = p["incumbent"] + p["challenger"] + p["abstain"]
        o[order] = {k: v / tot for k, v in p.items()} if tot > 0 else p
    return float(np.mean([o["fwd"]["challenger"], o["rev"]["challenger"]]))


def eligible(pred):
    return all(max(pred[o], key=pred[o].get) == "challenger" for o in ("fwd", "rev"))


def load_direct(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for det in payload["pair_details"]:
        key = (det["domain"], det["qid"])
        rows[key] = {"baseline": det["baseline"], "pairs": det["pairs"]}
    return rows


def load_condition(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {}
    for p in payload["pairs"]:
        rows.setdefault((p["domain"], p["qid"]), []).append(p)
    return rows


def domain_macro(per_query):
    per = {}
    for r in per_query:
        per.setdefault(r["domain"], []).append(r["policy"])
    return float(np.mean([np.mean(v) for v in per.values()]))


def paired_bootstrap(per_query, draws=10000, seed=20261120):
    rng = np.random.default_rng(seed)
    by_dom = {}
    for r in per_query:
        by_dom.setdefault(r["domain"], []).append(r["policy"] - r["baseline"])
    vals = []
    for _ in range(draws):
        acc = []
        for d in DOMAINS:
            v = np.array(by_dom.get(d, [0.0]))
            idx = rng.integers(0, len(v), len(v))
            acc.append(float(v[idx].mean()))
        vals.append(float(np.mean(acc)))
    arr = np.array(vals)
    return {
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "p_le_0": float((arr <= 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:/RETECO/github_repo")
    ap.add_argument("--direct", default="results/replacement_v2/exp_direct_probs_dev.json")
    ap.add_argument("--condition", default="results/replacement_v2/exp_condition_pairs_dev.json")
    ap.add_argument("--mode", choices=["both", "either", "intersect_strict"], default="both")
    ap.add_argument("--out", default="results/replacement_v2/ensemble_rules_dev.json")
    args = ap.parse_args()

    root = Path(args.root)
    direct = load_direct(root / args.direct)
    cond = load_condition(root / args.condition)

    per_query = []
    stats = {"swaps": 0, "helpful": 0, "harmful": 0, "neutral": 0}
    for key, det in direct.items():
        cond_pred = {(p["slot_rank"], p["challenger"]): p["pred"] for p in cond.get(key, [])}
        best = None
        for p in det["pairs"]:
            cp = cond_pred.get((p["slot_rank"], p["challenger"]))
            if cp is None:
                continue
            mr, mc = merged(p["pred"]), merged(cp)
            er, ec = eligible(p["pred"]), eligible(cp)
            ok = (er and ec) if args.mode == "both" else (er or ec)
            if not ok or max(mr, mc) <= 0.5:
                continue
            score = min(mr, mc) if args.mode == "both" else max(mr, mc)
            if best is None or score > best[0]:
                best = (score, p["delta_ndcg"], p["slot_rank"], p["challenger"])
        policy = det["baseline"]
        if best is not None:
            policy += best[1]
            stats["swaps"] += 1
            if best[1] > 1e-9:
                stats["helpful"] += 1
            elif best[1] < -1e-9:
                stats["harmful"] += 1
            else:
                stats["neutral"] += 1
        per_query.append({"domain": key[0], "qid": key[1], "baseline": det["baseline"], "policy": policy})

    macro = domain_macro(per_query)
    baseline = domain_macro([{**r, "policy": r["baseline"]} for r in per_query])
    boot = paired_bootstrap(per_query)
    out = {
        "mode": args.mode,
        "macro_policy": macro,
        "macro_baseline": baseline,
        "macro_delta": macro - baseline,
        "bootstrap": boot,
        "stats": stats,
        "n_queries": len(per_query),
        "per_query": per_query,
    }
    dest = root / (args.out if args.mode == "both" else args.out.replace(".json", f"_{args.mode}.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "per_query"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
