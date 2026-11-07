#!/usr/bin/env python3
"""Depth-consistent candidate sweep for one domain/split.

For every k in --k this reports, using exactly the top-k candidates:

  * BM25:  Recall@k, candidate-oracle nDCG@10
  * dense: Recall@k, candidate-oracle nDCG@10
  * union: Recall@k of |(A@k u B@k) & gold|, candidate-oracle nDCG@10
  * RRF:   reciprocal-rank fusion of the two full lists, truncated to k,
           scored for real (nDCG@10), plus its Recall@k and candidate-oracle

The union is a 2k-budget diagnostic; RRF is the equal-budget (k) fusion.

Invariant enforced for every list and every k: no relevant candidate => oracle 0.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "official"))
from ir_metrics import ndcg_at_k  # noqa: E402

from lib_reteco import load_qrels, read_trec_run  # noqa: E402

RRF_C = 60


def recall_at(ranked, gold, k):
    if not gold:
        return None
    return len(set(ranked[:k]) & gold) / len(gold)


def oracle(ranked, gold, k=10):
    gold_in = [d for d in ranked if d in gold]
    rest = [d for d in ranked if d not in gold]
    return ndcg_at_k(gold_in + rest, gold, k)


def rrf(a, b, k, c=RRF_C):
    """Reciprocal-rank fusion of two ranked lists, truncated to k."""
    score = {}
    for lst in (a, b):
        for r, d in enumerate(lst[:k], start=1):
            score[d] = score.get(d, 0.0) + 1.0 / (c + r)
    return sorted(score, key=lambda d: (-score[d], d))[:k]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def check(q, name, cand, orc, gold):
    if not (set(cand) & gold) and orc != 0.0:
        raise AssertionError(
            f"depth inconsistency: q={q} list={name} oracle={orc} with no gold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["iota", "bitcoin"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--root", default=os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--run-a", default=None, help="default: bm25_pyserini")
    ap.add_argument("--run-b", default=None, help="default: dense_bge")
    ap.add_argument("--k", type=int, nargs="+", default=[20, 50, 100, 200, 1000])
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    root = args.root.replace("\\", "/").rstrip("/")
    data = f"{root}/data/track1_tempo/{args.domain}"
    run_a = args.run_a or f"{root}/runs/bm25_pyserini_{args.domain}_{args.split}.trec"
    run_b = args.run_b or f"{root}/runs/dense_bge_{args.domain}_{args.split}.trec"

    qrels = load_qrels(data, args.split)
    A = read_trec_run(run_a)
    B = read_trec_run(run_b)

    rows = {}
    for q in qrels:
        gold = set(qrels[q])
        a_full = [d for _, _, d in A.get(q, [])]
        b_full = [d for _, _, d in B.get(q, [])]
        per_k = {}
        for k in args.k:
            a, b = a_full[:k], b_full[:k]
            union = a + [d for d in b if d not in set(a)]
            fused = rrf(a_full, b_full, k)
            rec = {
                "A": recall_at(a, gold, k),
                "B": recall_at(b, gold, k),
                "U": (len(set(union) & gold) / len(gold)) if gold else None,
                "RRF": recall_at(fused, gold, k),
            }
            orc = {
                "A": oracle(a, gold),
                "B": oracle(b, gold),
                "U": oracle(union, gold),
                "RRF": oracle(fused, gold),
            }
            for name, cand in (("A", a), ("B", b), ("U", union), ("RRF", fused)):
                check(q, f"{name}@{k}", cand, orc[name], gold)
            per_k[str(k)] = {
                "recall": rec,
                "oracle": orc,
                "rrf_ndcg@10": ndcg_at_k(fused, gold, 10),
                "gold": len(gold),
            }
        rows[q] = per_k

    out = {
        "domain": args.domain,
        "split": args.split,
        "run_a": os.path.abspath(run_a),
        "run_b": os.path.abspath(run_b),
        "num_topics": len(rows),
        "rrf_c": RRF_C,
        "by_k": {},
        "per_query": rows,
    }
    for k in args.k:
        kk = str(k)
        out["by_k"][kk] = {
            "Recall@k_A": mean([rows[q][kk]["recall"]["A"] for q in rows]),
            "Recall@k_B": mean([rows[q][kk]["recall"]["B"] for q in rows]),
            "Recall@k_Union": mean([rows[q][kk]["recall"]["U"] for q in rows]),
            "Recall@k_RRF": mean([rows[q][kk]["recall"]["RRF"] for q in rows]),
            "oracle_nDCG@10_A": mean([rows[q][kk]["oracle"]["A"] for q in rows]),
            "oracle_nDCG@10_B": mean([rows[q][kk]["oracle"]["B"] for q in rows]),
            "oracle_nDCG@10_Union": mean([rows[q][kk]["oracle"]["U"] for q in rows]),
            "oracle_nDCG@10_RRF": mean([rows[q][kk]["oracle"]["RRF"] for q in rows]),
            "RRF_nDCG@10": mean([rows[q][kk]["rrf_ndcg@10"] for q in rows]),
        }
    out["consistency_ok"] = True
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"# {args.domain}/{args.split}  ({out['num_topics']} topics, RRF c={RRF_C})")
    print(f"{'k':>5} | {'BM25 rec':>8} {'orc':>6} | {'dense rec':>9} {'orc':>6} | "
          f"{'union rec':>9} {'orc':>6} | {'RRF nDCG':>8} {'rec':>6} {'orc':>6}")
    for k in args.k:
        r = out["by_k"][str(k)]
        print(f"{k:>5} | {r['Recall@k_A']:>8.4f} {r['oracle_nDCG@10_A']:>6.4f} | "
              f"{r['Recall@k_B']:>9.4f} {r['oracle_nDCG@10_B']:>6.4f} | "
              f"{r['Recall@k_Union']:>9.4f} {r['oracle_nDCG@10_Union']:>6.4f} | "
              f"{r['RRF_nDCG@10']:>8.4f} {r['Recall@k_RRF']:>6.4f} "
              f"{r['oracle_nDCG@10_RRF']:>6.4f}")


if __name__ == "__main__":
    main()
