"""Does the facet pool add gold that the dense pool misses, especially for the 138
zero-score dev queries?"""
import json, os, sys
from collections import defaultdict
ROOT = os.path.expanduser("~/RETECO")
DOMAINS = ["bitcoin","cardano","economics","genealogy","history","hsm","iota","law",
           "monero","politics","quant","travel","workplace"]

def read_run(path, depth=None):
    d = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 4: continue
            q, doc, rank = p[0], p[2], int(float(p[3]))
            if depth is None or rank <= depth:
                d[q][doc] = rank
    return {q: dict(sorted(v.items(), key=lambda x: x[1])) for q, v in d.items()}

def load_qrels(path):
    rel = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(float(p[3])) > 0: rel[p[0]].add(p[2])
    return rel

tot = dict(q=0, dense=0, facet=0, union=0, zero=0, zd=0, zf=0, zu=0, new_gold=0)
per = {}
for d in DOMAINS:
    base = os.path.join(ROOT, "data/track1_tempo", d)
    qrels = load_qrels(os.path.join(base, "qrels_dev.txt"))
    dense = read_run(os.path.join(ROOT, "runs", f"dense_mxbai_{d}_dev.trec"), 100)
    facet = read_run(os.path.join(ROOT, "runs", f"facetstep_{d}_dev.trec"), None)
    frozen = read_run(os.path.join(ROOT, "runs", f"sub_sft_e2bc_{d}_dev.trec"), 10)
    s = dict(q=0, dense=0, facet=0, union=0, zero=0, zd=0, zf=0, zu=0, new_gold=0)
    for qid, g in qrels.items():
        if not g: continue
        s["q"] += 1
        dn = set(dense.get(qid, {})); fa = set(facet.get(qid, {}))
        top10 = set(frozen.get(qid, {}))
        zero = not (g & top10)
        hd, hf, hu = bool(g & dn), bool(g & fa), bool(g & (dn | fa))
        s["dense"] += hd; s["facet"] += hf; s["union"] += hu
        s["new_gold"] += len((g & fa) - dn)
        if zero:
            s["zero"] += 1; s["zd"] += hd; s["zf"] += hf; s["zu"] += hu
    per[d] = s
    for k in tot: tot[k] += s[k]
print(f'{"domain":11s} {"q":>4s} {"dense":>6s} {"facet":>6s} {"union":>6s} | {"zero":>4s} {"zd":>3s} {"zf":>3s} {"zu":>3s}')
for d, s in per.items():
    print(f'{d:11s} {s["q"]:4d} {s["dense"]:6d} {s["facet"]:6d} {s["union"]:6d} | {s["zero"]:4d} {s["zd"]:3d} {s["zf"]:3d} {s["zu"]:3d}')
print(f'{"TOTAL":11s} {tot["q"]:4d} {tot["dense"]:6d} {tot["facet"]:6d} {tot["union"]:6d} | {tot["zero"]:4d} {tot["zd"]:3d} {tot["zf"]:3d} {tot["zu"]:3d}')
print("new gold docs from facet pool:", tot["new_gold"])
json.dump({"per_domain": per, "total": tot}, open(os.path.join(ROOT, "results/analysis/pool_union_dev.json"), "w"), indent=1)
print("POOL_UNION_DONE")
