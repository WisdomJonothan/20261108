"""Independent rebuild of the frozen candidate pool from the raw runs.

Rule being audited (candidates_dev.json -> protocol.candidate_rule):
  union(dense top20, CE top20) minus frozen top10;
  order by best rank (tie: dense before CE, then docid); keep top 3.

Inputs read here are the committed TREC runs only (dense_mxbai, fusion100_full_ce_s17,
sub_sft_e2bc head10) plus candidates_dev.json for comparison. No qrels are read.
"""
import json, os
from collections import defaultdict

ROOT = r"D:\RETECO\github_repo"
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]


def read_run(path):
    order = defaultdict(list)
    seen = defaultdict(set)
    for ln in open(path, encoding="utf-8"):
        p = ln.split()
        if len(p) < 4:
            continue
        q, doc = p[0], p[2]
        if doc in seen[q]:
            continue
        seen[q].add(doc)
        order[q].append(doc)
    return order


def rebuild():
    got = {}
    for d in DOMAINS:
        dense = read_run(os.path.join(ROOT, "runs", f"dense_mxbai_{d}_dev.trec"))
        ce = read_run(os.path.join(ROOT, "runs", f"fusion100_full_ce_s17_20261115_{d}_dev_ce.trec"))
        frozen = read_run(os.path.join(ROOT, "runs", f"sub_sft_e2bc_{d}_dev_head10.trec"))
        for qid, fz in frozen.items():
            fz10 = fz[:10]
            d20 = dense.get(qid, [])[:20]
            c20 = ce.get(qid, [])[:20]
            dr = {doc: i + 1 for i, doc in enumerate(d20)}
            cr = {doc: i + 1 for i, doc in enumerate(c20)}
            cand = {}
            for doc in set(dr) | set(cr):
                best = min(dr.get(doc, 10 ** 6), cr.get(doc, 10 ** 6))
                src = 0 if dr.get(doc, 10 ** 6) <= cr.get(doc, 10 ** 6) else 1  # dense wins ties
                cand[doc] = (best, src)
            for doc in fz10:
                cand.pop(doc, None)
            ranked = sorted(cand.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]))
            got[(d, qid)] = dict(frozen_top10=fz10,
                                 candidates=[doc for doc, _ in ranked[:3]],
                                 pool_size=len(cand))
    return got


def main():
    got = rebuild()
    ref = json.load(open(os.path.join(ROOT, "results/replacement_v2/candidates_dev.json")))
    bad_frozen = bad_cand = 0
    pool_sizes = []
    checked = 0
    for row in ref["queries"]:
        d, qid = row["domain"], row["qid"]
        g = got[(d, qid)]
        checked += 1
        pool_sizes.append(g["pool_size"])
        if g["frozen_top10"] != row["frozen_top10"]:
            bad_frozen += 1
            if bad_frozen <= 3:
                print("FROZEN MISMATCH", d, qid)
        mine = g["candidates"]
        theirs = [c["docid"] if isinstance(c, dict) else c for c in row.get("candidates", [])]
        if mine != theirs:
            bad_cand += 1
            if bad_cand <= 5:
                print("CAND MISMATCH", d, qid, "mine", mine, "committed", theirs)
    print(f"checked {checked} queries; frozen_top10 mismatches {bad_frozen}; "
          f"candidate-list mismatches {bad_cand}")
    print(f"pool size before cap: min {min(pool_sizes)} max {max(pool_sizes)} "
          f"mean {sum(pool_sizes)/len(pool_sizes):.2f}")
    return bad_frozen, bad_cand


if __name__ == "__main__":
    raise SystemExit(0 if sum(main()) == 0 else 1)
