import json, math, os
from collections import defaultdict
ROOT = os.path.expanduser("~/RETECO")
D = ["bitcoin","cardano","economics","genealogy","history","hsm","iota","law","monero","politics","quant","travel","workplace"]
def rr(p):
    d = defaultdict(list)
    for line in open(p, encoding="utf-8"):
        q = line.split()
        if len(q) >= 4: d[q[0]].append((int(float(q[3])), q[2]))
    return {k: [x[1] for x in sorted(v)] for k, v in d.items()}
def qr(p):
    rel = defaultdict(dict)
    for line in open(p, encoding="utf-8"):
        x = line.split()
        if len(x) >= 4 and int(float(x[3])) > 0: rel[x[0]][x[2]] = int(float(x[3]))
    return rel
rows = []
for d in D:
    p = os.path.join(ROOT, "runs", f"union_zero_{d}_ce.trec")
    if not os.path.exists(p): continue
    run = rr(p); gold = qr(os.path.join(ROOT, "data/track1_tempo", d, "qrels_dev.txt"))
    for qid, docs in run.items():
        g = gold.get(qid, {})
        if not g: continue
        dcg = sum(g.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(docs[:10]))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), 10)))
        nd = dcg / idcg if idcg else 0.0
        hit = [doc for doc in docs[:10] if doc in g]
        rows.append({"domain": d, "qid": qid, "ce_ndcg10": round(nd, 4), "gold_in_ce_top10": len(hit)})
rows.sort(key=lambda r: -r["ce_ndcg10"])
print(f'{"domain":11s} {"qid":14s} {"ce_nDCG10":>9s} {"gold_in_top10":>13s}')
for r in rows: print(f'{r["domain"]:11s} {r["qid"]:14s} {r["ce_ndcg10"]:9.4f} {r["gold_in_ce_top10"]:13d}')
n = len(rows); fixed = sum(1 for r in rows if r["ce_ndcg10"] > 0)
print(f"queries {n}  CE-top10 contains gold {fixed}  mean CE nDCG@10 {sum(r['ce_ndcg10'] for r in rows)/n:.4f}")
json.dump(rows, open(os.path.join(ROOT, "results/analysis/zero_facet_ce.json"), "w"), indent=1)
