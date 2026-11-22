"""Depth ladder: for the zero-scoring dev queries, where does the first judged-relevant
document sit in the dense run (1000 deep by default)?"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/track1_tempo"
RUNS = ROOT / "runs"
DEPTHS = (10, 20, 50, 100, 200, 500, 1000)


def read_run(path, depth):
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 4:
            out.setdefault(p[0], []).append((p[2], float(p[3])))
    for q in out:
        out[q].sort(key=lambda x: -x[1])
        out[q] = [d for d, _ in out[q][:depth]]
    return out


def main():
    cand = json.loads((ROOT / "results/replacement_v2/candidates_dev.json").read_text(encoding="utf-8"))
    domains = sorted({r["domain"] for r in cand["queries"]})
    qrels, dense, ce = {}, {}, {}
    for d in domains:
        g = {}
        for ln in (DATA / d / "qrels_dev.txt").read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) >= 4 and int(p[3]) > 0:
                g.setdefault(p[0], set()).add(p[2])
        qrels[d] = g
        dense[d] = read_run(RUNS / f"dense_mxbai_{d}_dev.trec", 1000)
        ce[d] = read_run(RUNS / f"fusion100_full_ce_s17_20261115_{d}_dev_ce.trec", 100)

    zeros = []
    for r in cand["queries"]:
        d, q = r["domain"], str(r["qid"])
        g = qrels[d].get(q, set())
        if any(x in g for x in r["frozen_top10"]):
            continue
        ranked = dense[d].get(q, [])
        first = next((i + 1 for i, doc in enumerate(ranked) if doc in g), None)
        cerr = ce[d].get(q, [])
        cfirst = next((i + 1 for i, doc in enumerate(cerr) if doc in g), None)
        zeros.append({"domain": d, "qid": q, "dense_first_rel_rank": first, "ce_first_rel_rank": cfirst})

    ladder = {}
    for k in DEPTHS:
        ladder[str(k)] = sum(1 for z in zeros if z["dense_first_rel_rank"] and z["dense_first_rel_rank"] <= k)
    print("zero queries:", len(zeros))
    print("dense recall of a relevant document by depth:", ladder)
    print("ce top-100 has one:", sum(1 for z in zeros if z["ce_first_rel_rank"]))
    out = {"n_zero": len(zeros), "depth_ladder": ladder, "per_query": zeros}
    (ROOT / "results/replacement_v2/zero_depth_dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
