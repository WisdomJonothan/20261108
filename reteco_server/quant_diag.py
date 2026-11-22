"""Quant CE health check: score shapes, ties, overlap, recall of the CE run."""
import json, sys
from collections import defaultdict
sys.path.insert(0, "scripts")
from lib_reteco import load_qrels, read_trec_run

dom = "quant"
q = load_qrels(f"data/track1_tempo/{dom}", "dev")
dense = read_trec_run(f"runs/dense_mxbai_{dom}_dev.trec")
ce = read_trec_run(f"runs/frozenrepro_{dom}_dev_ce.trec")

def ndcg(run, k=10):
    import math
    tot = []
    for qid, gold in q.items():
        lst = run.get(qid, [])
        if not lst: tot.append(0.0); continue
        ideal = sorted(gold.values(), reverse=True)[:k]
        dcg = sum((2**gold.get(d,0)-1)/math.log2(i+2) for i,(_r,_s,d) in enumerate(lst[:k]))
        idcg = sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
        tot.append(dcg/idcg if idcg else 0.0)
    return sum(tot)/len(tot)

print("queries", len(q), "dense", round(ndcg(dense),4), "ce", round(ndcg(ce),4))
for qid, gold in q.items():
    dl = [x[2] for x in dense.get(qid,[])][:10]
    cl = [x[2] for x in ce.get(qid,[])][:10]
    cs = [x[1] for x in ce.get(qid,[])][:10]
    hits_d = len([d for d in dl if d in gold])
    hits_c = len([d for d in cl if d in gold])
    span = (round(min(cs),4), round(max(cs),4)) if cs else None
    uniq = len(set(round(s,4) for s in cs))
    print(f"{qid:20s} gold {len(gold):2d} dense_hit10 {hits_d} ce_hit10 {hits_c} ce_top10_uniq_scores {uniq}/{len(cs)} span {span}")
