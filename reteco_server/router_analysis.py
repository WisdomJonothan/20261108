"""Per-query router analysis: fused ranking vs CE-only ranking on the 138 zero queries.

Cheap, CPU-only. Reads the frozen submission runs and the cached CE scores for the
dense top-100 pool, then asks:
  1. how many of the 138 zero queries would gain from ranking by CE alone;
  2. what an oracle router (best of the two per query) would buy;
  3. whether a cheap feature (CE top1-top2 margin, dense-fused top1-top2 margin)
     separates the queries where CE-only wins.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]


def read_run(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 4:
            continue
        qid, docid = p[0], p[2]
        score = float(p[4]) if len(p) >= 6 else float(p[3])
        out.setdefault(qid, []).append((docid, score))
    for v in out.values():
        v.sort(key=lambda x: -x[1])
    return out


def read_qrels(path):
    gold = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 3:
            continue
        qid = p[0]
        docid = p[1] if len(p) == 3 else p[2]
        if float(p[-1]) > 0:
            gold.setdefault(qid, set()).add(docid)
    return gold


def ndcg10(ranked, gold):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(10, len(gold))))
    return dcg / idcg if idcg else 0.0


def margin(ranked):
    if len(ranked) < 2:
        return 0.0
    lo, hi = ranked[-1][1], ranked[0][1]
    rng = hi - lo
    return (ranked[0][1] - ranked[1][1]) / rng if rng > 0 else 0.0


rows = []
for d in DOMAINS:
    frozen = read_run(ROOT / "runs" / f"sub_sft_e2bc_{d}_dev.trec")
    ce = read_run(ROOT / "runs" / f"fusion100_full_ce_s17_20261115_{d}_dev_ce.trec")
    qrels = read_qrels(ROOT / "data" / "track1_tempo" / d / "qrels_dev.txt")
    for qid in frozen:
        gold = qrels.get(qid, set())
        fz = [doc for doc, _ in frozen[qid][:10]]
        ce_only = [doc for doc, _ in ce.get(qid, [])[:10]]
        if not ce_only:
            continue
        rows.append({
            "domain": d, "qid": qid,
            "frozen": ndcg10(fz, gold), "ce": ndcg10(ce_only, gold),
            "ce_margin": round(margin(ce[qid]), 6),
            "frozen_margin": round(margin(frozen[qid][:100]), 6),
        })

n = len(rows)
zero = [r for r in rows if r["frozen"] == 0.0]
print(f"queries {n}; zero under frozen {len(zero)}")
for name, sel in (("all", rows), ("zeros", zero)):
    fz = sum(r["frozen"] for r in sel) / len(sel)
    ce = sum(r["ce"] for r in sel) / len(sel)
    best = sum(max(r["frozen"], r["ce"]) for r in sel) / len(sel)
    print(f"{name:6s} frozen {fz:.4f}  ce-only {ce:.4f}  per-query-best {best:.4f}  ({len(sel)} q)")
gained = [r for r in zero if r["ce"] > 0]
print(f"zeros fixed by CE-only ranking: {len(gained)} / {len(zero)}")
if gained:
    print("  example fixes:", [(r["qid"], round(r["ce"], 3)) for r in gained[:5]])

# cheap router: where does CE-only win, by CE margin decile?
wins = [r for r in rows if r["ce"] > r["frozen"]]
loss = [r for r in rows if r["ce"] < r["frozen"]]
print(f"CE-only better on {len(wins)} q, worse on {len(loss)} q")
for label, sel in (("wins", wins), ("losses", loss)):
    if sel:
        print(f"  {label:7s} mean CE margin {sum(r['ce_margin'] for r in sel)/len(sel):.3f}"
              f"  mean frozen margin {sum(r['frozen_margin'] for r in sel)/len(sel):.3f}")

out = ROOT / "results" / "replacement_v2" / "router_dev.json"
out.write_text(json.dumps({"n": n, "zeros": len(zero), "rows": rows}, indent=1), encoding="utf-8")
print("wrote", out)
