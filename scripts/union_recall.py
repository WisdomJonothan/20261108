#!/usr/bin/env python3
"""Candidate-set analysis: BM25 vs dense and their union — depth-consistent.

Every number for a given k uses exactly the top-k candidates of that k:

  * Recall@k of each run and of the candidate-set union |(A@k u B@k) & gold|/|gold|
  * candidate oracle nDCG@10, i.e. the ideal re-ranking of exactly those top-k
    candidates (upper bound any re-ranker restricted to these candidates can reach)
  * overlap@k between the two lists

Consistency invariant: if no relevant document is inside the candidate set, the
candidate oracle must be exactly 0. The script asserts this for every query.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "official"))
from ir_metrics import ndcg_at_k  # noqa: E402

from lib_reteco import load_qrels, read_trec_run  # noqa: E402


def recall_at(ranked, gold, k):
    if not gold:
        return None
    return len(set(ranked[:k]) & gold) / len(gold)


def oracle(ranked, gold, k=10):
    """Ideal re-ranking of exactly the given candidate list."""
    gold_in = [d for d in ranked if d in gold]
    rest = [d for d in ranked if d not in gold]
    return ndcg_at_k(gold_in + rest, gold, k)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    k = args.k

    qrels = load_qrels(args.data, args.split)
    A = read_trec_run(args.run_a)
    B = read_trec_run(args.run_b)

    rows = {}
    for q in qrels:
        gold = set(qrels[q])
        a = [d for _, _, d in A.get(q, [])][:k]
        b = [d for _, _, d in B.get(q, [])][:k]
        union = a + [d for d in b if d not in set(a)]

        row = {
            "gold": len(gold),
            "candidate_depth": k,
            "recA": recall_at(a, gold, k),
            "recB": recall_at(b, gold, k),
            "recU": (len(set(union) & gold) / len(gold)) if gold else None,
            "oracleA": oracle(a, gold),
            "oracleB": oracle(b, gold),
            "oracleU": oracle(union, gold),
            "overlap": len(set(a) & set(b)) / max(1, min(k, len(a), len(b))),
        }
        # depth consistency: no relevant candidates -> oracle must be zero
        for name, cand, orc in (("A", a, row["oracleA"]),
                                ("B", b, row["oracleB"]),
                                ("U", union, row["oracleU"])):
            if not (set(cand) & gold) and orc != 0.0:
                raise AssertionError(
                    f"depth inconsistency: q={q} list={name} has no gold in top-{k} "
                    f"but oracle={orc}")
        rows[q] = row

    out = {
        "run_a": os.path.abspath(args.run_a),
        "run_b": os.path.abspath(args.run_b),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "candidate_depth": k,
        "num_topics": len(rows),
        f"Recall@{k}_A": mean([r["recA"] for r in rows.values()]),
        f"Recall@{k}_B": mean([r["recB"] for r in rows.values()]),
        f"Recall@{k}_Union": mean([r["recU"] for r in rows.values()]),
        f"oracle_nDCG@10_A_cand{k}": mean([r["oracleA"] for r in rows.values()]),
        f"oracle_nDCG@10_B_cand{k}": mean([r["oracleB"] for r in rows.values()]),
        f"oracle_nDCG@10_Union_cand{k}": mean([r["oracleU"] for r in rows.values()]),
        f"overlap@{k}": mean([r["overlap"] for r in rows.values()]),
        "consistency_ok": True,
        "per_query": rows,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({kk: vv for kk, vv in out.items() if kk != "per_query"},
                     indent=2))


if __name__ == "__main__":
    main()
