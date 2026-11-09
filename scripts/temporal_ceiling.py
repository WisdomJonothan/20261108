#!/usr/bin/env python3
"""Ceiling analysis: how much could a perfect time-consistency reranker add?

For every anchored dev query we take the frozen dense top-K, sort the candidates
by a *gold-free* temporal criterion (time-consistent before time-inconsistent,
stable inside each block, unknown time last) and recompute nDCG@10. The delta
against the dense run is an upper bound for any reranker that only uses
query-anchor vs document-year consistency -- it is an oracle over the ranking,
but it uses no qrels to decide the order.

Anchors come from the raw query text + guidance summary/events/anchors; document
time is the same high-confidence extraction used by hardneg_mine.py.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_examples, load_qrels, read_trec_run  # noqa: E402
from hardneg_mine import doc_time, load_guidance, query_interval, gap  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ndcg_at_10(order, gold):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(order[:10]) if d in gold)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), 10)))
    return dcg / idcg if idcg else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ddir = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    ids, docs = load_corpus(ddir)
    docs_by_id = dict(zip(ids, docs))
    qids, queries = load_examples(ddir, args.split)
    gold = {q: set(d) for q, d in load_qrels(ddir, args.split).items()}
    guid = load_guidance(ddir, args.split)
    run = read_trec_run(args.run)

    rows = []
    for qid in qids:
        iv = query_interval(queries[qids.index(qid)], guid.get(qid, {}))
        cand = [d for _r, _s, d in run.get(qid, [])][: args.k]
        base = ndcg_at_10(cand, gold.get(qid, set()))
        if iv is None or not cand:
            rows.append({"qid": qid, "anchored": 0, "base": base, "perfect": base})
            continue
        cons, inc, unk = [], [], []
        for d in cand:
            dt = doc_time(docs_by_id.get(d, ""))
            if dt is None:
                unk.append(d)
            elif gap(dt["years"], iv) == 0:
                cons.append(d)
            else:
                inc.append(d)
        order = cons + inc + unk
        demote = [d for d in cand if d in set(cons) or d in set(unk)] + inc
        rows.append({"qid": qid, "anchored": 1, "base": base,
                     "perfect": ndcg_at_10(order, gold.get(qid, set())),
                     "demote_only": ndcg_at_10(demote, gold.get(qid, set())),
                     "n_cons": len(cons), "n_inc": len(inc), "n_unk": len(unk),
                     "inc_rate": len(inc) / max(1, len(cons) + len(inc))})

    anc = [r for r in rows if r["anchored"]]
    summary = {
        "domain": args.domain, "split": args.split, "run": args.run, "k": args.k,
        "queries": len(rows), "anchored": len(anc),
        "dense_macro": float(np.mean([r["base"] for r in rows])),
        "perfect_macro": float(np.mean([r["perfect"] for r in rows])),
        "dense_macro_anchored": float(np.mean([r["base"] for r in anc])) if anc else 0.0,
        "perfect_macro_anchored": float(np.mean([r["perfect"] for r in anc])) if anc else 0.0,
        "demote_macro": float(np.mean([r.get("demote_only", r["base"]) for r in rows])),
        "demote_macro_anchored": float(np.mean([r.get("demote_only", r["base"]) for r in anc])) if anc else 0.0,
        "mean_inc_rate": float(np.mean([r["inc_rate"] for r in anc])) if anc else 0.0,
        "queries_with_inc": int(sum(1 for r in anc if r["n_inc"] > 0)),
        "mean_inc_per_query": float(np.mean([r["n_inc"] for r in anc])) if anc else 0.0,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "queries": rows}, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
