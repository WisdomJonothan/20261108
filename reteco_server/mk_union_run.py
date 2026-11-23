import json, os
from collections import defaultdict
ROOT = os.path.expanduser("~/RETECO")
rows = json.load(open(os.path.join(ROOT, "results/analysis/zero_facet_gain.json")))
bydom = defaultdict(list)
for r in rows: bydom[r["domain"]].append(r["qid"])
def rr(p):
    d = defaultdict(list)
    for line in open(p, encoding="utf-8"):
        q = line.split()
        if len(q) >= 4: d[q[0]].append((int(float(q[3])), q[2]))
    return d
os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
tot = 0
with open(os.path.join(ROOT, "runs/union_zero_dev.trec"), "w", encoding="utf-8") as out:
    for dom, qids in bydom.items():
        dense = rr(os.path.join(ROOT, "runs", f"dense_mxbai_{dom}_dev.trec"))
        facet = rr(os.path.join(ROOT, "runs", f"facetstep_{dom}_dev.trec"))
        for qid in qids:
            seen, merged = set(), []
            for r, doc in sorted(dense.get(qid, []))[:100]:
                if doc not in seen: seen.add(doc); merged.append(doc)
            for r, doc in sorted(facet.get(qid, [])):
                if doc not in seen: seen.add(doc); merged.append(doc)
            for i, doc in enumerate(merged, 1):
                out.write(f"{qid} Q0 {doc} {i} {-i:.4f} union\n")
            tot += 1
print("union queries", tot)
