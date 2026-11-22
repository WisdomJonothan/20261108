"""Independent reproduction of the frozen baseline and per-domain table
from RAW TREC runs + qrels (does not touch the study's JSON artefacts)."""
import json, math, sys
from pathlib import Path

ROOT = Path("D:/RETECO/github_repo")
DOM = ["bitcoin","cardano","economics","genealogy","history","hsm","iota",
       "law","monero","politics","quant","travel","workplace"]

def read_run(p):
    out = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        f = ln.split()
        qid, docid, rank = f[0], f[2], int(f[3])
        out.setdefault(qid, []).append((rank, docid))
    for q in out: out[q].sort()
    return {q: [d for _, d in v] for q, v in out.items()}

def qrels(d):
    g = {}
    for ln in (ROOT/"data/track1_tempo"/d/"qrels_dev.txt").read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 4 and int(p[3]) > 0: g.setdefault(p[0], set()).add(p[2])
    return g

def ndcg(ranked, gold, k=10):
    dcg = sum(1.0/math.log2(i+1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    ideal = sum(1.0/math.log2(i+1) for i in range(1, min(k, len(gold))+1))
    return dcg/ideal if ideal else 0.0

def main(tag, split="dev_head10"):
    per_dom, allq = {}, []
    for d in DOM:
        run = read_run(ROOT/"runs"/f"{tag}_{d}_{split}.trec")
        g = qrels(d)
        vals = []
        for qid, gold in g.items():            # iterate QRELS -> missing queries score 0
            vals.append(ndcg(run.get(qid, []), gold))
            allq.append((d, qid, vals[-1]))
        missing = [q for q in g if q not in run]
        per_dom[d] = (sum(vals)/len(vals), len(vals), len(missing))
    dom_macro = sum(v[0] for v in per_dom.values())/len(DOM)
    topic_macro = sum(x[2] for x in allq)/len(allq)
    print(f"tag={tag}")
    for d,(m,n,miss) in per_dom.items(): print(f"  {d:10s} n={n:3d} missing={miss:2d} ndcg={m:.6f}")
    print(f"domain_macro = {dom_macro:.10f}")
    print(f"topic_macro  = {topic_macro:.10f}")
    return per_dom, dom_macro, topic_macro

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sub_sft_e2bc", sys.argv[2] if len(sys.argv)>2 else "dev_head10")
