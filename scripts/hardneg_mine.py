#!/usr/bin/env python3
"""Hard-negative temporal discrimination + gated rerank (Phase 8).

Implements the pre-registered design in results/HARDNEG_SPEC.md:

  * candidates: frozen dense top-K (score >= top1 - 0.10) [+ BM25 top-100]
  * query interval: four-digit years in the raw query text and in the guidance
    fields query_summary / query_temporal_events / key_time_anchors
    (retrieval_plan and gold_passage_annotations are NEVER read)
  * document time: years in the title and the first 300 tokens, kept only when
    high confidence (one year, or span <= 3, or dominant year >= 2 mentions and
    span <= 10)
  * time-hard: all document years outside the query interval +/- 1, and gap >= 2
  * topic-hard: topic cosine >= 0.80 (query with year expressions removed) or
    >= 2 shared non-year terms with idf >= 4

Two outputs:
  1. --mine  : hard-negative pairs (jsonl) + per-query counts
  2. --score : 8-feature table per pair (jsonl) for the discriminator

The script never uses qrels to build features; qrels are only read by --eval.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_examples, load_qrels, read_jsonl, read_trec_run  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
HTML = re.compile(r"<[^>]+>")
TOKEN = re.compile(r"[a-z0-9]+")


def years_in(text):
    return sorted({int(y) for y in YEAR.findall(text or "")})


def strip_years(text):
    return YEAR.sub(" ", text or "")


def load_guidance(ddir, split):
    p = os.path.join(ddir, f"guidance_{split}.jsonl")
    if not os.path.isfile(p):
        return {}
    out = {}
    for r in read_jsonl(p):
        q = r.get("query_guidance") or {}
        keep = {k: q.get(k) for k in
                ("query_summary", "query_temporal_events", "key_time_anchors",
                 "expected_granularity", "is_temporal_query")
                if q.get(k) is not None}
        out[r["id"]] = keep
    return out


def query_interval(query, guid):
    txt = [query]
    for k in ("query_summary", "query_temporal_events", "key_time_anchors"):
        v = (guid or {}).get(k)
        if isinstance(v, str):
            txt.append(v)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    txt.append(x)
                elif isinstance(x, dict):
                    txt.extend(str(y) for y in x.values() if isinstance(y, (str, int)))
    ys = years_in("\n".join(txt))
    if not ys:
        return None
    raw = "\n".join(txt)
    # a range like "2015-2017" / "2015 to 2017" widens the interval
    if len(ys) >= 2 and re.search(r"\d{4}\s*(?:-|–|to|and)\s*\d{4}", raw):
        return (min(ys), max(ys))
    return (ys[-1], ys[-1])


def doc_time(text):
    """High-confidence document time from title + first 300 tokens."""
    body = HTML.sub(" ", text or "")
    head = body[:1200]
    ys = years_in(head)
    if not ys:
        return None
    counts = {}
    for y in YEAR.findall(head):
        counts[int(y)] = counts.get(int(y), 0) + 1
    span = max(ys) - min(ys)
    if len(ys) == 1:
        return {"years": ys, "conf": 1.0}
    if span <= 3:
        return {"years": ys, "conf": 0.8}
    dom = max(counts, key=counts.get)
    if counts[dom] >= 2 and span <= 10:
        return {"years": [dom], "conf": 0.6}
    if span <= 10:
        # multi-year but bounded: keep the span as a low-confidence interval
        return {"years": ys, "conf": 0.4}
    return None


def gap(doc_years, interval):
    lo, hi = interval
    if any(lo - 1 <= y <= hi + 1 for y in doc_years):
        return 0
    return min(abs(y - lo) if y < lo else abs(y - hi) for y in doc_years)


def load_run(path, k):
    out = {}
    for qid, rows in read_trec_run(path).items():
        for rank, _score, did in rows:
            if rank <= k:
                out.setdefault(qid, []).append(did)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", required=True, help="dense TREC run (top-1000)")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--score-drop", type=float, default=0.10)
    ap.add_argument("--min-gap", type=int, default=2)
    ap.add_argument("--topic-cos", type=float, default=0.80)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default="")
    args = ap.parse_args()

    ddir = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    ids, docs = load_corpus(ddir)
    qids, queries = load_examples(ddir, args.split)
    guid = load_guidance(ddir, args.split)
    pos = {q: set(d.keys()) for q, d in load_qrels(ddir, args.split).items()}

    # dense scores per query (for the top1 - 0.10 window)
    scores = {}
    with open(args.run, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 6:
                continue
            # TREC: qid Q0 docid rank score run_id  -> score is p[4], NOT p[3]
            q, d, s = p[0], p[2], float(p[4])
            scores.setdefault(q, {})[d] = s
    ranked = load_run(args.run, args.k)
    docs_by_id = dict(zip(ids, docs))

    pairs, stats = [], []
    for qid in qids:
        q = queries[qids.index(qid)]
        iv = query_interval(q, guid.get(qid, {}))
        if iv is None:
            stats.append({"qid": qid, "anchored": 0, "interval": None})
            continue
        cand = ranked.get(qid, [])[: args.k]
        if not cand:
            stats.append({"qid": qid, "anchored": 1, "interval": iv, "cand": 0})
            continue
        top1 = max(scores.get(qid, {}).get(d, 0.0) for d in cand)
        pool = [d for d in cand
                if scores.get(qid, {}).get(d, 0.0) >= top1 - args.score_drop]
        mism = []
        for d in pool:
            if d in pos.get(qid, set()):
                continue
            dt = doc_time(docs_by_id.get(d, ""))
            if dt is None:
                continue
            g = gap(dt["years"], iv)
            if g >= args.min_gap:
                mism.append({"qid": qid, "docid": d, "gap": g,
                             "doc_years": dt["years"], "doc_conf": dt["conf"],
                             "interval": list(iv)})
        for m in mism:
            pairs.append(m)
        stats.append({"qid": qid, "anchored": 1, "interval": list(iv),
                      "cand": len(cand), "pool": len(pool), "hard_neg": len(mism)})

    with open(args.out, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1)
    nq = sum(1 for s in stats if s["anchored"])
    print(json.dumps({"domain": args.domain, "split": args.split,
                      "queries": len(qids), "anchored": nq,
                      "pairs": len(pairs),
                      "queries_with_negs": sum(1 for s in stats if s.get("hard_neg"))},
                     indent=1))


if __name__ == "__main__":
    main()
