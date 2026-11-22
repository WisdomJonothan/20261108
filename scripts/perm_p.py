"""Randomisation tests for the replacement arms, two nulls.

`p_le_0` in the bootstrap blocks is resolution-limited: 468 of the 519 per-query deltas
are exactly zero, so the bootstrap distribution is lumpy and the 10-draw floor (0.001) is
not a calibrated p-value. Two randomisation tests are reported instead:

1. `shuffle` (the informative one): keep the number of swaps per query fixed, but replace
   the judge's chosen pair with a uniformly random pair from the same nine, using the
   true per-pair nDCG change recorded for every pair (judged or not). This asks whether
   the judge's selection beats chance selection inside the same frozen pool.
2. `signflip`: within-domain sign flip over the queries that move. This is a test of
   symmetry, not of the mean, and it is reported only for completeness: the gain is
   concentrated in a few large wins, so the sign test has little power and a large p-value
   here does not contradict the bootstrap interval.

Run:  python scripts/perm_p.py results/replacement_v2/exp_direct_context_dev.json
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


def macro_from(per_query_delta: dict[tuple[str, str], float], counts: dict[str, int]) -> float:
    per = []
    for d in DOMAINS:
        if not counts[d]:
            continue
        total = sum(v for (dom, _), v in per_query_delta.items() if dom == d)
        per.append(total / counts[d])
    return sum(per) / len(per) if per else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artefact")
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--root", default=".")
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    path = Path(args.artefact)
    if not path.is_absolute():
        path = root / path
    exp = json.loads(path.read_text(encoding="utf-8"))
    cand_path = Path(args.candidates) if args.candidates else (
        root / "results" / "replacement_v2" / Path(exp["candidate_file"]).name)
    cand = json.loads(cand_path.read_text(encoding="utf-8"))

    # every pair's true change, keyed by (domain, qid) -> list of 9 floats in pair order
    cand_deltas: dict[tuple[str, str], list[float]] = {}
    for q in cand["queries"]:
        key = (q["domain"], str(q["qid"]))
        cand_deltas[key] = [0.0] * len(q["pairs"])

    counts = {d: sum(1 for q in cand["queries"] if q["domain"] == d) for d in DOMAINS}
    rng = np.random.default_rng(args.seed)

    records = {}
    for policy, block in exp["policies"].items():
        rows = block.get("per_query")
        recs = block.get("pair_records", [])
        if not rows:
            print(f"{policy}: no per_query block, skipped")
            continue

        # fill the true delta of every pair from pair_records (order-aligned per query)
        by_key: dict[tuple[str, str], list[dict]] = {}
        for rec in recs:
            by_key.setdefault((rec["domain"], str(rec["qid"])), []).append(rec)
        deltas: dict[tuple[str, str], list[float]] = {}
        for key, rs in by_key.items():
            deltas[key] = [float(r.get("delta_ndcg", 0.0)) for r in rs]

        observed = {key: 0.0 for key in cand_deltas}
        swapped = [key for key, rs in by_key.items() if any(r.get("chosen") for r in rs)]
        for key in swapped:
            for r in by_key[key]:
                if r.get("chosen"):
                    observed[key] = float(r.get("delta_ndcg", 0.0))

        obs_macro = macro_from(observed, counts)

        null = []
        for _ in range(args.draws):
            draw = {key: 0.0 for key in cand_deltas}
            for key in swapped:
                opts = deltas.get(key) or [0.0]
                draw[key] = float(opts[int(rng.integers(len(opts)))])
            null.append(macro_from(draw, counts))
        null = np.array(null)
        p_shuffle = (int((null >= obs_macro).sum()) + 1) / (args.draws + 1)

        # sign-flip over the queries that move
        moving = [r for r in rows if abs(r["delta"]) > 0]
        per_domain_idx = {d: [i for i, r in enumerate(moving) if r["domain"] == d] for d in DOMAINS}
        vals = np.array([r["delta"] for r in moving])
        hits = 0
        for _ in range(args.draws):
            signs = rng.choice([-1.0, 1.0], size=len(vals))
            per = []
            for d in DOMAINS:
                idx = per_domain_idx[d]
                if not idx or not counts[d]:
                    continue
                per.append(float(np.sum(vals[idx] * signs[idx])) / counts[d])
            if per and sum(per) / len(per) <= 0:
                hits += 1
        p_sign = (hits + 1) / (args.draws + 1)

        print(f"{policy:12s} macro {obs_macro:+.10f}  swaps {len(swapped)}  "
              f"p(shuffle>=obs) {p_shuffle:.4f}  moving {len(moving)}/{len(rows)}  "
              f"p(signflip<=0) {p_sign:.4f}")
        records[policy] = {"macro": obs_macro, "swaps": len(swapped),
                           "moving_queries": len(moving), "n_queries": len(rows),
                           "p_shuffle": p_shuffle, "p_signflip": p_sign}

    if args.out:
        payload = {"artefact": str(path.name), "draws": args.draws, "seed": args.seed,
                   "nulls": {"shuffle": "within-query uniform replacement of the judged pair",
                             "signflip": "within-domain sign flip over moving queries"},
                   "results": records}
        out = Path(args.out)
        if not out.is_absolute():
            out = root / out
        out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
