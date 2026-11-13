#!/usr/bin/env python3
"""Counterfactual coverage bound for dev.

For every dev query we hold the *relevance pattern* of the top ten fixed: the
same number of judged-relevant documents sits at exactly the same ranks, so
nDCG@10 cannot change.  We then ask which relevant documents could have filled
those slots, choosing them from the same candidate pool (the dense top-100 that
the cross-encoder sees).  The union of annotated steps they cover is the
coverage a ranker that cared about coverage -- and nothing else -- could get
at zero nDCG cost.  Comparing that with what the deployed system covers says
whether the remaining coverage gap is a *ranking* problem (same pool, same
relevance budget) or a *pool* problem.

Reported per domain, domain-macro:

    cov_now      steps covered by the fused top ten (the system as deployed)
    cov_cf       best step coverage with the same relevant-slot pattern
    cov_pool     steps reachable anywhere in the dense top-100 (pool ceiling)
    cov_all      steps reachable anywhere in the qrels (annotation ceiling)

Also: how many queries gain, the size of the gain, and how often a query has
one relevant slot where the picked document covers fewer steps than another
document available at the same slot.

    python scripts/coverage_cf_bound.py --runs runs/sub_sft_e2bc_{domain}_dev.trec \
        --pool runs/dense_mxbai_{domain}_dev.trec --out results/COVERAGE_CF_BOUND.json
"""
import argparse
import itertools
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]


def read_qrels(path):
    gold = defaultdict(set)
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) >= 4 and int(p[3]) > 0:
            gold[p[0]].add(p[2])
    return gold


def read_steps(path):
    steps = defaultdict(lambda: defaultdict(set))
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) < 4 or int(p[3]) <= 0:
            continue
        qid, docid = p[0], p[2]
        if "_step" not in qid:
            continue
        base, step = qid.rsplit("_step", 1)
        steps[base][step].add(docid)
    return steps


def read_run(path, depth=None):
    out = defaultdict(list)
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) >= 6:
            out[p[0]].append((int(p[3]), p[2]))
    for q in out:
        out[q].sort()
        if depth:
            out[q] = out[q][:depth]
    return out


def ndcg10(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def best_union(docs, doc_steps, k):
    """Largest union of steps any k of `docs` can cover."""
    if k <= 0 or not docs:
        return 0
    if k == 1:
        return max(len(doc_steps.get(d, ())) for d in docs)
    best = 0
    for combo in itertools.combinations(docs, k):
        u = set()
        for d in combo:
            u |= doc_steps.get(d, set())
        if len(u) > best:
            best = len(u)
            if best == sum(len(doc_steps.get(d, ())) for d in combo):
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/sub_sft_e2bc_{domain}_dev.trec")
    ap.add_argument("--pool", default="runs/dense_mxbai_{domain}_dev.trec")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "track1_tempo"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    per_domain = {}
    macro = defaultdict(list)
    for d in DOMAINS:
        run_p = args.runs.format(domain=d)
        pool_p = args.pool.format(domain=d)
        if not (os.path.isfile(run_p) and os.path.isfile(pool_p)):
            print("[skip] %s" % d)
            continue
        gold = read_qrels(os.path.join(args.data, d, "qrels_dev.txt"))
        steps = read_steps(os.path.join(args.data, d, "qrels_steps_dev.txt"))
        run = read_run(run_p)
        pool = read_run(pool_p, depth=100)

        n = 0
        cov_now = cov_cf = cov_pool = cov_all = 0.0
        gain_q = 0
        gain_steps = 0
        loss_slot_q = 0   # one relevant slot, a complementary doc was available
        nd_sum = 0.0
        for q in sorted(run):
            ranked = [docid for _r, docid in run[q]]
            top = ranked[:args.k]
            g_all = gold.get(q, set())
            pool_docs = [docid for _r, docid in pool.get(q, [])]
            pool_set = set(pool_docs)
            g_pool = g_all & pool_set
            st = steps.get(q, {})

            def doc_steps(doc):
                return {s for s, docs in st.items() if doc in docs}

            cur_docs = [x for x in top if x in g_all]
            cur = set()
            for x in cur_docs:
                cur |= doc_steps(x)
            # best k relevant docs from the pool
            cf = best_union(g_pool, {x: doc_steps(x) for x in g_pool}, len(cur_docs))
            pool_steps = set()
            for x in g_pool:
                pool_steps |= doc_steps(x)
            all_steps = set(st)
            n += 1
            cov_now += len(cur)
            cov_cf += cf
            cov_pool += len(pool_steps)
            cov_all += len(all_steps)
            nd_sum += ndcg10(ranked, g_all, args.k)
            if cf > len(cur):
                gain_q += 1
                gain_steps += cf - len(cur)
            if len(cur_docs) == 1:
                best_one = max((len(doc_steps(x)) for x in g_pool), default=0)
                if best_one > len(cur):
                    loss_slot_q += 1

        per_domain[d] = {
            "queries": n,
            "cov_now": round(cov_now / n, 4),
            "cov_cf": round(cov_cf / n, 4),
            "cov_pool": round(cov_pool / n, 4),
            "cov_all": round(cov_all / n, 4),
            "gain_queries": gain_q,
            "gain_steps": gain_steps,
            "single_slot_left_on_table": loss_slot_q,
            "ndcg10": round(nd_sum / n, 4),
        }
        for key in ("cov_now", "cov_cf", "cov_pool", "cov_all", "ndcg10"):
            macro[key].append(per_domain[d][key])

    agg = {k: round(sum(v) / len(v), 4) for k, v in macro.items()}
    gain_q = sum(x["gain_queries"] for x in per_domain.values())
    gain_s = sum(x["gain_steps"] for x in per_domain.values())
    single = sum(x["single_slot_left_on_table"] for x in per_domain.values())
    out = {
        "runs": args.runs, "pool": args.pool, "k": args.k,
        "domain_macro": agg,
        "totals": {"queries": sum(x["queries"] for x in per_domain.values()),
                   "gain_queries": gain_q, "gain_steps": gain_s,
                   "single_slot_left_on_table": single},
        "domains": per_domain,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("domain_macro", "totals")}, indent=1))


if __name__ == "__main__":
    main()
