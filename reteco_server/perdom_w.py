"""Per-domain fusion weight fitted on the 969 train queries (leak-free fold CE runs),
then applied to dev with the frozenrepro CE run. Reports dev domain-macro."""
import math, sys, json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/homeC/wujiaheng/RETECO")
DOMAINS = ["bitcoin","cardano","economics","genealogy","history","hsm","iota","law",
           "monero","politics","quant","travel","workplace"]
GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

def read_run(p):
    out = defaultdict(dict)
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            qid, _, doc, rank, score, _ = ln.split()
            out[qid][doc] = float(score)
    return {q: d for q, d in out.items()}

def read_qrels(p):
    out = defaultdict(dict)
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            p_ = ln.split()
            if len(p_) < 4:
                continue
            out[p_[0]][p_[2]] = int(p_[3])
    return out

def ndcg_at_10(ranking, gold):
    if not ranking:
        return 0.0
    ideal = sorted(gold.values(), reverse=True)[:10]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        return 0.0
    dcg = sum((2**gold.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranking[:10]))
    return dcg / idcg

def zscores(scores):
    v = list(scores.values())
    if len(v) < 2:
        return {d: 0.0 for d in scores}
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5 or 1.0
    return {d: (x - m) / sd for d, x in scores.items()}

def fused_ndcg(dense_q, ce_q, gold, w, k=100):
    a = dict(list(dense_q.items())[:k])
    b = dict(list(ce_q.items())[:k])
    za, zb = zscores(a), zscores(b)
    docs = set(za) | set(zb)
    sc = {d: za.get(d, -3.0) + w * zb.get(d, -3.0) for d in docs}
    ranking = sorted(sc, key=lambda d: (-sc[d], d))
    return ndcg_at_10(ranking, gold)

# ---- train: per-domain best w using out-of-fold CE runs
print("domain      n   w*    train_ndcg@w*   train@w=1")
best_w = {}
for d in DOMAINS:
    qrels = read_qrels(ROOT / f"data/track1_tempo/{d}/qrels_train.txt")
    dense = read_run(ROOT / f"runs/dense_mxbai_{d}_train.trec")
    ce = {}
    for f in ("f0", "f1", "f2"):
        p = ROOT / f"runs/{f}_ce_{d}_train_ce.trec"
        if p.exists():
            ce.update(read_run(p))
    qids = [q for q in qrels if q in dense and q in ce]
    if not qids:
        print(f"{d:12s} no train data"); best_w[d] = 1.0; continue
    per_w = {}
    for w in GRID:
        vals = [fused_ndcg(dense[q], ce[q], qrels[q], w) for q in qids]
        per_w[w] = sum(vals) / len(vals)
    bw = max(per_w, key=per_w.get)
    best_w[d] = bw
    print(f"{d:12s} {len(qids):3d} {bw:4.2f}   {per_w[bw]:.4f}        {per_w[1.0]:.4f}")

# ---- dev: apply per-domain w* and compare against w=1 everywhere
print()
print("dev: w* per domain vs frozen w=1 (frozenrepro CE)")
tot, tot1 = [], []
for d in DOMAINS:
    qrels = read_qrels(ROOT / f"data/track1_tempo/{d}/qrels_dev.txt")
    dense = read_run(ROOT / f"runs/dense_mxbai_{d}_dev.trec")
    ce = read_run(ROOT / f"runs/frozenrepro_{d}_dev_ce.trec")
    vals = [fused_ndcg(dense[q], ce[q], qrels[q], best_w[d]) for q in qrels if q in dense and q in ce]
    vals1 = [fused_ndcg(dense[q], ce[q], qrels[q], 1.0) for q in qrels if q in dense and q in ce]
    m, m1 = sum(vals)/len(vals), sum(vals1)/len(vals1)
    tot.append(m); tot1.append(m1)
    print(f"{d:12s} w*={best_w[d]:4.2f} dev {m:.4f}  w=1 {m1:.4f}  delta {m-m1:+.4f}")
print()
print(f"domain-macro  fitted {sum(tot)/len(tot):.6f}   w=1 {sum(tot1)/len(tot1):.6f}   delta {sum(tot)/len(tot)-sum(tot1)/len(tot1):+.6f}")
json.dump({"best_w": best_w, "dev_fitted": sum(tot)/len(tot), "dev_w1": sum(tot1)/len(tot1)},
          open(ROOT/"results/replacement_v2/per_domain_w.json","w"), indent=2)
