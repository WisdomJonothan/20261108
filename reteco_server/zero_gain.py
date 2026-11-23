"""Which zero-score dev queries does the facet pool newly cover, and can the frozen CE
put one of those documents into the top-10?"""
import json, os
from collections import defaultdict
ROOT = os.path.expanduser("~/RETECO")
D = ["bitcoin","cardano","economics","genealogy","history","hsm","iota","law","monero","politics","quant","travel","workplace"]
def rr(p, depth=None):
    d = defaultdict(dict)
    for line in open(p, encoding="utf-8"):
        q = line.split()
        if len(q) < 4: continue
        r = int(float(q[3]))
        if depth is None or r <= depth: d[q[0]][q[2]] = r
    return d
def qr(p):
    rel = defaultdict(set)
    for line in open(p, encoding="utf-8"):
        x = line.split()
        if len(x) >= 4 and int(float(x[3])) > 0: rel[x[0]].add(x[2])
    return rel
rows = []
for d in D:
    b = os.path.join(ROOT, "data/track1_tempo", d)
    gold = qr(os.path.join(b, "qrels_dev.txt"))
    dense = rr(os.path.join(ROOT, "runs", f"dense_mxbai_{d}_dev.trec"), 100)
    facet = rr(os.path.join(ROOT, "runs", f"facetstep_{d}_dev.trec"))
    frozen = rr(os.path.join(ROOT, "runs", f"sub_sft_e2bc_{d}_dev.trec"), 10)
    for qid, g in gold.items():
        if g & set(frozen.get(qid, {})):   # not a zero query
            continue
        in_dense = g & set(dense.get(qid, {}))
        in_facet = g & set(facet.get(qid, {}))
        new = in_facet - in_dense
        if new:
            rows.append({"domain": d, "qid": qid, "n_gold": len(g),
                         "gold_in_dense": len(in_dense), "new_from_facet": sorted(new),
                         "pool_size": len(set(dense.get(qid, {})) | set(facet.get(qid, {})))})
json.dump(rows, open(os.path.join(ROOT, "results/analysis/zero_facet_gain.json"), "w"), indent=1)
print("queries newly covered:", len(rows))
for r in rows: print(r["domain"], r["qid"], "gold", r["n_gold"], "new", len(r["new_from_facet"]), "pool", r["pool_size"])
