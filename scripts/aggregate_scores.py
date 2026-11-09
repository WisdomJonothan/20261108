#!/usr/bin/env python3
"""Aggregate per-domain `score_run.py` JSONs into the two macros that matter.

RETECO's official Track-1a aggregation is nDCG@10 **per query, averaged within
each domain, then averaged over the 13 domains** (domain-macro). The topic-macro
(flat mean over all 519 queries) is a diagnostic that over-weights history (240
queries) and under-weights iota (3). Both are reported here, plus per-domain
rows, so a table can never silently switch between them.

Usage:
  python aggregate_scores.py --pattern "results/score_*_dev.json" --tag fused_1_1 \
      --json results/aggregate_fused_1_1_dev.json
"""
import argparse
import glob
import json
import os

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRIC = "nDCG@10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", nargs="+", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--metric", default=METRIC)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    paths = []
    for pat in args.pattern:
        paths.extend(sorted(glob.glob(os.path.join(ROOT, pat))))
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if METRIC not in d:
            continue
        domain = os.path.basename(d.get("data", "")).strip()
        rows.append({
            "domain": domain or os.path.basename(p),
            "n": d.get("num_topics", 0),
            "n_in_run": d.get("num_topics_in_run", 0),
            "ndcg10": d[METRIC],
            "file": os.path.relpath(p, ROOT),
        })
    if not rows:
        raise SystemExit("no score files matched")
    rows.sort(key=lambda r: r["domain"])
    # the official number silently changes if a per-domain score file is
    # missing or duplicated, so make the 13-domain contract explicit
    if len(rows) != 13:
        raise SystemExit(f"expected 13 domains, matched {len(rows)}: "
                         + ", ".join(r["domain"] for r in rows))
    n_tot = sum(r["n"] for r in rows)
    topic_macro = sum(r["ndcg10"] * r["n"] for r in rows) / n_tot
    domain_macro = sum(r["ndcg10"] for r in rows) / len(rows)
    big = [r for r in rows if r["n"] >= 19]
    big_topic = sum(r["ndcg10"] * r["n"] for r in big) / sum(r["n"] for r in big)
    big_domain = sum(r["ndcg10"] for r in big) / len(big)

    print(f"{'domain':12s} {'n':>4s} {'nDCG@10':>8s}")
    for r in rows:
        print(f"{r['domain']:12s} {r['n']:>4d} {r['ndcg10']:>8.4f}")
    print("-" * 30)
    print(f"domains={len(rows)}  topics={n_tot}")
    print(f"topic-macro             = {topic_macro:.4f}")
    print(f"domain-macro (official) = {domain_macro:.4f}")
    print(f"topic-macro  (n>=19, {len(big)} domains, {sum(r['n'] for r in big)} q) = {big_topic:.4f}")
    print(f"domain-macro (n>=19)    = {big_domain:.4f}")

    out = {"tag": args.tag, "metric": METRIC, "domains": rows,
           "num_domains": len(rows), "num_topics": n_tot,
           "topic_macro": topic_macro, "domain_macro": domain_macro,
           "topic_macro_n_ge_19": big_topic, "domain_macro_n_ge_19": big_domain,
           "domains_n_ge_19": [r["domain"] for r in big]}
    if args.json:
        path = os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"-> {path}")


if __name__ == "__main__":
    main()
