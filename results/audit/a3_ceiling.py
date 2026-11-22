"""Independent recomputation of the frozen baseline and the one-swap oracle ceiling
from the raw TREC runs + qrels only (no committed pair_details, no candidates JSON).

Protocol: frozen top-10 from runs/sub_sft_e2bc_<d>_dev_head10.trec; candidates =
union(dense_mxbai top20, CE top20) minus frozen top10, ordered by best rank (dense
wins ties, then docid), cap 3; each candidate vs slots 8, 9, 10; at most one swap;
binary nDCG@10 over qrels with missing queries scoring 0; domain-macro = mean of 13
per-domain means.
"""
import json, math, os
from collections import defaultdict

ROOT = r"D:\RETECO\github_repo"
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]
K = 10


def read_run(path):
    o = defaultdict(list)
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) < 4:
            continue
        o[p[0]].append((int(p[3]), p[2]))
    for q in o:
        o[q].sort()
    return {q: [d for _, d in v] for q, v in o.items()}


def read_qrels(domain):
    rel = defaultdict(set)
    for ln in open(os.path.join(ROOT, f"data/track1_tempo/{domain}/qrels_dev.txt"), encoding="utf-8"):
        p = ln.split()
        if len(p) >= 4 and int(p[3]) > 0:
            rel[p[0]].add(p[2])
    return rel


def ndcg10(ranking, gold):
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranking[:K]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(K, len(gold))))
    return dcg / ideal if ideal else 0.0


def main():
    per_domain = {}
    n_beneficial = 0
    total_gain = 0.0
    for domain in DOMAINS:
        frozen = read_run(os.path.join(ROOT, "runs", f"sub_sft_e2bc_{domain}_dev_head10.trec"))
        dense = read_run(os.path.join(ROOT, "runs", f"dense_mxbai_{domain}_dev.trec"))
        ce = read_run(os.path.join(ROOT, "runs", f"fusion100_full_ce_s17_20261115_{domain}_dev_ce.trec"))
        qrels = read_qrels(domain)
        rows = []
        base_scores, oracle_scores = [], []
        for qid, gold in qrels.items():
            top = frozen[qid][:10]
            dr = {d: i + 1 for i, d in enumerate(dense.get(qid, [])[:20])}
            cr = {d: i + 1 for i, d in enumerate(ce.get(qid, [])[:20])}
            cands = []
            for doc in set(dr) | set(cr):
                if doc in top:
                    continue
                best = min(dr.get(doc, 10 ** 6), cr.get(doc, 10 ** 6))
                src = 0 if dr.get(doc, 10 ** 6) <= cr.get(doc, 10 ** 6) else 1
                cands.append((best, src, doc))
            cands.sort()
            cands = cands[:3]
            base = ndcg10(top, gold)
            best_delta = 0.0
            for _, _, doc in cands:
                for slot in (7, 8, 9):
                    ranking = list(top)
                    ranking[slot] = doc
                    d = ndcg10(ranking, gold) - base
                    if d > best_delta:
                        best_delta = d
            base_scores.append(base)
            oracle_scores.append(base + best_delta)
            if best_delta > 0:
                n_beneficial += 1
                total_gain += best_delta
        per_domain[domain] = (sum(base_scores) / len(base_scores),
                              sum(oracle_scores) / len(oracle_scores))
    base_macro = sum(v[0] for v in per_domain.values()) / len(per_domain)
    oracle_macro = sum(v[1] for v in per_domain.values()) / len(per_domain)
    allbase = [v[0] for v in per_domain.values()]
    print("domain            n     base     oracle")
    for d in DOMAINS:
        print(f"{d:12s}  --  {per_domain[d][0]:.6f}  {per_domain[d][1]:.6f}")
    print(f"\nbaseline domain-macro  {base_macro:.10f}")
    print(f"oracle   domain-macro  {oracle_macro:.10f}")
    print(f"ceiling gain           {oracle_macro - base_macro:.10f}")
    print(f"queries with beneficial pair {n_beneficial} / {sum(1 for _ in DOMAINS) and 519}")
    print(f"sum of positive per-query gains {total_gain:.9f}")


if __name__ == "__main__":
    main()
