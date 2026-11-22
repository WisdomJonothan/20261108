"""Add a per_query block to a replacement experiment artefact.

The headline artefact `exp_direct_context_dev.json` stores pair-level records but not the
per-query nDCG values, so a reader can only replay the bootstrap by importing the replay
script. This tool recomputes, for every policy in the artefact, the baseline, policy and
delta nDCG@10 for each of the 519 dev queries from the frozen candidates plus the
committed qrels, and writes them back into the artefact under `per_query`.

Run:  python scripts/add_per_query.py results/replacement_v2/exp_direct_context_dev.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DOMAINS = [
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
    "law", "monero", "politics", "quant", "travel", "workplace",
]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """qid -> {docid: grade} for the positive judgements."""
    out: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        qid = parts[0]
        docid = parts[1] if len(parts) == 3 else parts[2]
        try:
            grade = int(float(parts[-1]))
        except ValueError:
            continue
        if grade > 0:
            out.setdefault(qid, {})[docid] = grade
    return out


def ndcg10(ranked: list[str], gold: dict[str, int]) -> float:
    dcg = 0.0
    for i, docid in enumerate(ranked[:10]):
        g = gold.get(docid, 0)
        if g:
            dcg += g / math.log2(i + 2)
    ideal = sorted(gold.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artefact")
    ap.add_argument("--root", default=".")
    ap.add_argument("--candidates", default=None,
                    help="defaults to the artefact's candidate_file, resolved under results/replacement_v2")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    path = Path(args.artefact)
    if not path.is_absolute():
        path = root / path
    exp = json.loads(path.read_text(encoding="utf-8"))

    cand_path = Path(args.candidates) if args.candidates else (
        root / "results" / "replacement_v2" / Path(exp["candidate_file"]).name
    )
    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    queries = cand["queries"]

    gold_by_domain = {d: read_qrels(root / "data" / "track1_tempo" / d / "qrels_dev.txt")
                      for d in DOMAINS}

    # pair_records carry no docids; they align by position with queries[*].pairs
    cand_by_key = {(q["domain"], str(q["qid"])): q for q in queries}

    for policy, block in exp["policies"].items():
        recs_by_key: dict[tuple[str, str], list[dict]] = {}
        for rec in block.get("pair_records", []):
            recs_by_key.setdefault((rec["domain"], str(rec["qid"])), []).append(rec)
        chosen: dict[tuple[str, str], str] = {}
        for key, recs in recs_by_key.items():
            pairs = cand_by_key[key]["pairs"]
            assert len(pairs) == len(recs), (key, len(pairs), len(recs))
            for pair, rec in zip(pairs, recs):
                if rec.get("chosen"):
                    chosen[key] = pair["incoming_docid"]
        rows = []
        for q in queries:
            dom = q["domain"]
            qid = str(q["qid"])
            gold = gold_by_domain[dom].get(qid, {})
            base = [str(d) for d in q["frozen_top10"]]
            ranked = list(base)
            key = (dom, qid)
            incoming = chosen.get(key)
            if incoming is not None:
                pairs = cand_by_key[key]["pairs"]
                recs = recs_by_key[key]
                for pair, rec in zip(pairs, recs):
                    if rec.get("chosen"):
                        ranked[int(rec["slot_rank"]) - 1] = pair["incoming_docid"]
                        break
            b = ndcg10(base, gold)
            p = ndcg10(ranked, gold)
            rows.append({"domain": dom, "qid": qid, "baseline": b, "policy": p, "delta": p - b})

        per_domain = {}
        for d in DOMAINS:
            vals = [r["delta"] for r in rows if r["domain"] == d]
            if vals:
                per_domain[d] = sum(vals) / len(vals)
        macro = sum(per_domain.values()) / len(per_domain)
        block["per_query"] = rows
        block["per_query_macro"] = macro
        print(f"{policy:12s} queries {len(rows)}  macro {macro:.10f}  "
              f"artefact {block.get('macro_delta', float('nan')):+.10f}")

    path.write_text(json.dumps(exp, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
