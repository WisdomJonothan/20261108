#!/usr/bin/env python3
"""Score a Track 1a run with the RETECO official metrics + Phase-0 diagnostics.

Official metric: nDCG@10 via the organizers' ir_metrics.ndcg_at_k
(vendored as scripts/ir_metrics.py from the official starter kit).

Diagnostics added here (NOT official leaderboard metrics):
  * Recall@{10,100,1000}
  * oracle@10_cand_all   -- ideal re-ranking of the candidates stored in the run
                            file (the file holds the top-1000 by default)
  * oracle@10_cand100    -- ideal re-ranking restricted to the top-100 candidates
                            (what a re-ranker fed with 100 candidates can reach)
  * TP@10 / TR@10 / TC@10 -- official temporal diagnostics, using steps_*.jsonl
                   as the required-period definition (as scorer.py does)

Usage:
  python score_run.py --run X.trec --data D:/RETECO/data/track1_tempo/iota \
      --split dev --json out.json
"""
import argparse
import json
import os
import sys
from math import log2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_metrics import (ndcg_at_k, temporal_precision_at_k,  # noqa: E402
                        temporal_relevance_at_k)

from lib_reteco import load_qrels, read_trec_run  # noqa: E402


def load_periods(domain_dir, split):
    path = os.path.join(domain_dir, f"steps_{split}.jsonl")
    periods = {}
    if not os.path.isfile(path):
        return periods
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            periods[rec["id"]] = [set(st.get("gold_ids", [])) for st in rec.get("steps", [])]
    return periods


def dcg(rels):
    return sum(r / log2(i + 2) for i, r in enumerate(rels))


def ndcg_ref(ranked, gold, k=10):
    """Independent re-implementation used only to cross-check ir_metrics."""
    grades = [1.0 if d in gold else 0.0 for d in ranked[:k]]
    ideal = [1.0] * min(len(gold), k)
    return dcg(grades) / dcg(ideal) if ideal else 0.0


def recall_at(ranked, gold, k):
    if not gold:
        return None
    return len(set(ranked[:k]) & gold) / len(gold)


def oracle_ndcg(ranked, gold, k=10):
    """nDCG@k after perfectly re-ordering the retrieved candidates."""
    cand = [d for d in ranked]
    gold_in = [d for d in cand if d in gold]
    rest = [d for d in cand if d not in gold]
    return ndcg_at_k(gold_in + rest, gold, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--json", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow-missing", action="store_true",
                    help="score only the topics present in the run (default: count "
                         "missing topics as 0, which is what the leaderboard does)")
    args = ap.parse_args()

    qrels = load_qrels(args.data, args.split)
    run = read_trec_run(args.run)
    periods = load_periods(args.data, args.split)

    qids = [q for q in qrels if q in run]
    missing = [q for q in qrels if q not in run]

    per_query = {}
    for q in qids:
        ranked = [d for _, _, d in run[q]]
        gold = set(qrels[q])
        nd = ndcg_at_k(ranked, gold, 10)
        nd_ref = ndcg_ref(ranked, gold, 10)
        assert abs(nd - nd_ref) < 1e-12, (q, nd, nd_ref)
        row = {
            "nDCG@10": nd,
            "Rec@10": recall_at(ranked, gold, 10),
            "Rec@100": recall_at(ranked, gold, 100),
            "Rec@1000": recall_at(ranked, gold, 1000),
            # candidate oracle at the depth actually stored in the run file
            "oracle@10_cand_all": oracle_ndcg(ranked, gold, 10),
            # ceiling of a re-ranker that only sees the top-100 candidates
            "oracle@10_cand100": oracle_ndcg(ranked[:100], gold, 10),
            "TR@10": temporal_relevance_at_k(ranked, gold, 10),
            "TP@10": temporal_precision_at_k(ranked, gold, 10),
            "num_gold": len(gold),
            "num_candidates": len(ranked),
        }
        # depth consistency: a candidate set without relevance cannot support an oracle
        for key, cand in (("oracle@10_cand_all", ranked),
                          ("oracle@10_cand100", ranked[:100])):
            if not (set(cand) & gold):
                assert row[key] == 0.0, (q, key, row[key])
        pg = periods.get(q)
        if pg:
            top = set(ranked[:10])
            row["TC@10"] = sum(1 for gs in pg if top & set(gs)) / len(pg)
        per_query[q] = row

    if not args.allow_missing:
        zero = {k: 0.0 for k in ("nDCG@10", "Rec@10", "Rec@100", "Rec@1000",
                                "oracle@10_cand_all", "oracle@10_cand100",
                                "TR@10", "TP@10", "TC@10")}
        for q in missing:
            row = dict(zero)
            row["num_gold"] = len(qrels[q])
            row["num_candidates"] = 0
            row["missing_topic"] = True
            per_query[q] = row

    def mean(key):
        vals = [r[key] for r in per_query.values() if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "tag": args.tag or os.path.basename(args.run),
        "run_file": os.path.abspath(args.run),
        "data": os.path.abspath(args.data),
        "split": args.split,
        "num_topics": len(qrels),
        "num_topics_in_run": len(qids),
        "topics_missing_from_run": sorted(missing),
        "missing_counted_as_zero": not args.allow_missing,
        "nDCG@10": mean("nDCG@10"),
        "Recall@10": mean("Rec@10"),
        "Recall@100": mean("Rec@100"),
        "Recall@1000": mean("Rec@1000"),
        "oracle_nDCG@10_cand_all": mean("oracle@10_cand_all"),
        "oracle_nDCG@10_cand100": mean("oracle@10_cand100"),
        "candidate_depth": max((r["num_candidates"] for r in per_query.values()),
                               default=None),
        "TP@10": mean("TP@10"),
        "TR@10": mean("TR@10"),
        "TC@10": mean("TC@10"),
        "per_query": per_query,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"# {summary['tag']}  ({len(qids)} topics, split={args.split})")
    for k in ["nDCG@10", "Recall@10", "Recall@100", "Recall@1000",
              "oracle_nDCG@10_cand100", "oracle_nDCG@10_cand_all",
              "TP@10", "TR@10", "TC@10"]:
        v = summary[k]
        print(f"  {k:14s} {v:.4f}" if v is not None else f"  {k:14s} n/a")
    if missing:
        print(f"  WARNING: {len(missing)} qrels topics missing from run")


if __name__ == "__main__":
    main()
