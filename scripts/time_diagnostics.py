#!/usr/bin/env python3
"""Step 3: is document-level temporal diagnosis feasible on RETECO Track 1a?

The released corpus has only (id, content) -- there is no publication-date
metadata field. So every temporal signal must be extracted from the text.
This script measures what is actually available, per domain:

  * share of documents with an explicit 4-digit year (1990-2026)
  * share with a month name / ISO-like date
  * the distribution of years (which content times the corpus talks about)
  * per-query (dev): guidance is_temporal_query flag and how many of the gold
    documents carry an explicit date (i.e. how much a text-extracted period
    filter could even see)

Nothing here feeds ranking; it is a feasibility measurement only.
"""
import argparse
import json
import os
import re
from collections import Counter

from lib_reteco import load_corpus, load_examples, load_qrels

YEAR = re.compile(r"\b(19[9]\d|20[0-2]\d)\b")
ISO = re.compile(r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b")
MONTHS = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.I)
RELATIVE = re.compile(r"\b(yesterday|today|tomorrow|last (week|month|year)|"
                      r"\d+ (days?|weeks?|months?|years?) ago|recently)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    ids, texts = load_corpus(args.data)
    n = len(texts)
    year_hits = {"any_year": 0, "iso_date": 0, "month_name": 0, "relative": 0}
    year_counter = Counter()
    per_doc_years = {}
    for i, t in enumerate(texts):
        ys = YEAR.findall(t)
        if ys:
            year_hits["any_year"] += 1
            year_counter.update(ys)
            per_doc_years[ids[i]] = ys
        if ISO.search(t):
            year_hits["iso_date"] += 1
        if MONTHS.search(t):
            year_hits["month_name"] += 1
        if RELATIVE.search(t):
            year_hits["relative"] += 1

    qrels = load_qrels(args.data, args.split)
    qids, queries = load_examples(args.data, args.split)

    guidance_path = os.path.join(args.data, f"guidance_{args.split}.jsonl")
    guidance = {}
    if os.path.isfile(guidance_path):
        with open(guidance_path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    r = json.loads(ln)
                    guidance[r["id"]] = r["query_guidance"]

    per_query = []
    for qid, q in zip(qids, queries):
        gold = list(qrels.get(qid, {}))
        g = guidance.get(qid, {})
        in_gold = [d for d in gold if d in per_doc_years]
        gold_years = Counter()
        for d in in_gold:
            gold_years.update(per_doc_years[d])
        per_query.append({
            "qid": qid,
            "query": q[:120],
            "num_gold": len(gold),
            "gold_with_year": len(in_gold),
            "gold_years": gold_years.most_common(5),
            "is_temporal_flag": g.get("is_temporal_query"),
            "granularity": g.get("expected_granularity"),
            "key_time_anchors": g.get("key_time_anchors", [])[:5],
        })

    out = {
        "domain_dir": os.path.abspath(args.data),
        "num_docs": n,
        "doc_fields": ["id", "content"],
        "pct_any_year": 100.0 * year_hits["any_year"] / n,
        "pct_iso_date": 100.0 * year_hits["iso_date"] / n,
        "pct_month_name": 100.0 * year_hits["month_name"] / n,
        "pct_relative": 100.0 * year_hits["relative"] / n,
        "year_distribution_top": year_counter.most_common(20),
        "split": args.split,
        "num_queries": len(qids),
        "num_queries_flagged_temporal": sum(
            1 for r in per_query if r["is_temporal_flag"]),
        "mean_gold_with_year_frac": (
            sum(r["gold_with_year"] / r["num_gold"] for r in per_query if r["num_gold"])
            / max(1, sum(1 for r in per_query if r["num_gold"]))),
        "per_query": per_query,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"# {os.path.basename(os.path.dirname(args.data.rstrip('/')))}"
          f"/{os.path.basename(args.data.rstrip('/'))}  {n} docs")
    print(f"  docs with any 4-digit year : {out['pct_any_year']:.1f}%")
    print(f"  docs with ISO-like date    : {out['pct_iso_date']:.1f}%")
    print(f"  docs with a month name     : {out['pct_month_name']:.1f}%")
    print(f"  docs with relative time    : {out['pct_relative']:.1f}%")
    print(f"  top years: {year_counter.most_common(10)}")
    print(f"  queries flagged temporal   : {out['num_queries_flagged_temporal']}"
          f"/{len(qids)}")
    print(f"  mean share of gold docs with a year: {out['mean_gold_with_year_frac']:.2f}")


if __name__ == "__main__":
    main()
