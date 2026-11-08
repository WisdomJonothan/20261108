#!/usr/bin/env python3
"""Feasibility study: can we build "same topic, different time" hard negatives?

For each query we take the dense top-K candidates (which are topic-relevant by
construction) and split them by a *content-time* criterion:

  * query time anchors: four-digit years mentioned in the raw query text plus
    years inside the guidance fields (summary / events / anchors);
  * document time: four-digit years in the document text.

A candidate is "time-mismatched" when it has no year within +/-1 of any query
anchor (documents with no year at all count as unknown, reported separately).

We report, per query and per domain: how many top-K candidates are
time-mismatched, how many of them are gold (which would make the label noisy),
and a few examples. This only measures feasibility; it is not a reranker.
"""
import argparse
import json
import os
import re
from collections import Counter

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, read_jsonl, read_trec_run  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def years_in(text):
    return sorted({int(y) for y in YEAR.findall(text or "")})


def query_anchors(query, guid):
    txt = [query]
    if guid:
        for k in ("query_summary", "query_temporal_events", "key_time_anchors"):
            v = guid.get(k)
            if isinstance(v, str):
                txt.append(v)
            elif isinstance(v, list):
                txt.extend(str(x) for x in v)
    return years_in("\n".join(txt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--top", type=int, default=50, help="topic-relevant window")
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ddir = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    ids, docs = load_corpus(ddir)
    doc_year = {i: years_in(t) for i, t in zip(ids, docs)}
    qmeta = {r["id"]: r for r in read_jsonl(os.path.join(ddir, f"examples_{args.split}.jsonl"))}
    guid = {}
    gpath = os.path.join(ddir, f"guidance_{args.split}.jsonl")
    if os.path.isfile(gpath):
        guid = {r["id"]: r.get("query_guidance", {}) for r in read_jsonl(gpath)}
    gold = {r["id"]: set(r.get("gold_ids", [])) for r in qmeta.values()}
    run = read_trec_run(os.path.join(ROOT, "runs", f"dense_bge_{args.domain}_{args.split}.trec"))

    per_query, agg = {}, Counter()
    for qid, ranked in run.items():
        if qid not in qmeta:
            continue
        anchors = query_anchors(qmeta[qid]["query"], guid.get(qid))
        cands = ranked[:args.top]
        mismatch, unknown, hit, gold_mismatch = [], [], [], []
        for _rank, _score, did in cands:
            ys = doc_year.get(did, [])
            if not ys:
                unknown.append(did)
                continue
            near = [y for y in ys if anchors and min(abs(y - a) for a in anchors) <= args.tolerance]
            (hit if near else mismatch).append(did)
            if not near and did in gold.get(qid, set()):
                gold_mismatch.append(did)
        per_query[qid] = {
            "anchors": anchors, "n_cands": len(cands),
            "matched": len(hit), "mismatched": len(mismatch), "no_year": len(unknown),
            "gold_total": len(gold.get(qid, [])),
            "gold_mismatched": len(gold_mismatch),
            "example_mismatch": mismatch[:3],
        }
        agg["queries"] += 1
        agg["q_with_anchor"] += bool(anchors)
        agg["q_with_mismatch"] += bool(mismatch) and bool(anchors)
        agg["q_with_gold_contradiction"] += bool(gold_mismatch)
    summary = {
        "domain": args.domain, "split": args.split, "k": args.top,
        "tolerance": args.tolerance, "counts": dict(agg),
        "mean_mismatched_per_query": round(
            sum(v["mismatched"] for v in per_query.values()) / max(1, len(per_query)), 2),
        "mean_no_year_per_query": round(
            sum(v["no_year"] for v in per_query.values()) / max(1, len(per_query)), 2),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "per_query": per_query}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
