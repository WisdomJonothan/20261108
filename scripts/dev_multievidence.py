#!/usr/bin/env python3
"""Multi-evidence diagnostics on dev runs (user memo m00374, innovation line A).

The official annotation marks *steps* of each question and, for every step, the
documents that evidence it. A reranker that only learns "find the single easiest
relevant passage" can score well on nDCG@10 while never covering a second step.
This script measures that directly, per domain, from run files that already
exist -- no model is loaded and nothing is re-scored.

Per query and per domain it reports:

  gold            number of judged-relevant documents in the qrels
  gold@10         how many of them the run puts in the top 10
  multi@10        does the top 10 contain >= 2 distinct relevant documents
  steps           steps that have at least one gold document in the pool
  steps@10        those covered by at least one top-10 document
  step_cov        steps@10 / steps
  step_recall     steps covered anywhere in the run (retrieval ceiling)
  gold_found      queries with >= 1 relevant document anywhere in the run
  ndcg@10         nDCG@10 under the official domain-macro convention

Aggregation is domain-macro (13 domains weighted equally), matching the
official metric, plus micro averages for reference.

    python scripts/dev_multievidence.py --runs runs/sub_sft11_<d>_dev.trec \
        --out results/DEV_MULTIEVIDENCE.json
"""
import argparse
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]


def read_qrels(path):
    gold = defaultdict(set)
    if not os.path.isfile(path):
        return gold
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) >= 4 and int(p[3]) > 0:
            gold[p[0]].add(p[2])
    return gold


def read_step_qrels(path):
    """{qid: {step_label: set(docid)}} from qid_stepK rows."""
    steps = defaultdict(lambda: defaultdict(set))
    if not os.path.isfile(path):
        return steps
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) < 4 or int(p[3]) <= 0:
            continue
        qid, _, docid = p[0], p[1], p[2]
        if "_step" not in qid:
            continue
        base, step = qid.rsplit("_step", 1)
        steps[base][step].add(docid)
    return steps


def read_run(path):
    out = defaultdict(list)
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) >= 6:
            out[p[0]].append((int(p[3]), p[2]))
    for q in out:
        out[q].sort()
    return out


def ndcg_at_k(ranked, gold, k=10):
    dcg = 0.0
    for i, d in enumerate(ranked[:k], 1):
        if d in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="template with {domain}, e.g. runs/sub_sft11_{domain}_dev.trec")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "track1_tempo"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    per_domain, pooled = {}, defaultdict(float)
    for d in DOMAINS:
        run_path = args.runs.format(domain=d)
        if not os.path.isfile(run_path):
            print("[skip] %s (no run)" % d)
            continue
        gold = read_qrels(os.path.join(args.data, d, "qrels_dev.txt"))
        steps = read_step_qrels(os.path.join(args.data, d, "qrels_steps_dev.txt"))
        run = read_run(run_path)
        qids = sorted(run)
        n = g10 = multi = 0
        step_tot = step_hit = step_found = 0
        gold_found = 0
        nd = []
        for q in qids:
            ranked = [docid for _r, docid in run[q]]
            top = ranked[:args.k]
            pool = set(ranked)
            g = gold.get(q, set())
            hit = len(set(top) & g)
            n += 1
            g10 += hit
            multi += int(hit >= 2)
            gold_found += int(bool(pool & g))
            nd.append(ndcg_at_k(ranked, g, args.k))
            for step, docs in steps.get(q, {}).items():
                if not docs:
                    continue
                step_tot += 1
                if set(top) & docs:
                    step_hit += 1
                if pool & docs:
                    step_found += 1
        per_domain[d] = {
            "queries": n,
            "gold_mean": round(g10 / max(1, n), 4),
            "multi_at_k": round(multi / max(1, n), 4),
            "steps": step_tot,
            "step_cov": round(step_hit / step_tot, 4) if step_tot else None,
            "step_recall": round(step_found / step_tot, 4) if step_tot else None,
            "gold_found": round(gold_found / max(1, n), 4),
            "ndcg10": round(sum(nd) / max(1, n), 4),
        }
        for key in ("gold_mean", "multi_at_k", "step_cov", "step_recall",
                    "gold_found", "ndcg10", "queries"):
            pooled[key] += per_domain[d].get(key) or 0.0

    macro = {}
    for key in ("gold_mean", "multi_at_k", "step_cov", "step_recall",
                "gold_found", "ndcg10"):
        vals = [v[key] for v in per_domain.values() if v.get(key) is not None]
        macro[key] = round(sum(vals) / len(vals), 4) if vals else None
    n_dom = len(per_domain)
    macro["queries"] = sum(v["queries"] for v in per_domain.values())
    n_tot = macro["queries"]
    # query-weighted micro means (pooled rates are already per-query averages,
    # so they must be weighted by the number of queries, not divided again)
    micro = {k: round(sum((v[k] or 0.0) * v["queries"]
                          for v in per_domain.values()) / n_tot, 4)
             for k in ("gold_mean", "multi_at_k", "gold_found", "ndcg10")} if n_tot else {}
    steps_tot = sum(v["steps"] for v in per_domain.values())
    if steps_tot:
        for key in ("step_cov", "step_recall"):
            micro[key] = round(sum((v[key] or 0.0) * v["steps"]
                                   for v in per_domain.values()) / steps_tot, 4)

    out = {"run_template": args.runs, "k": args.k, "domains": per_domain,
           "domain_macro": macro, "micro": micro}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"domain_macro": macro, "per_domain": {
        d: v for d, v in per_domain.items()}}, indent=1))


if __name__ == "__main__":
    main()
