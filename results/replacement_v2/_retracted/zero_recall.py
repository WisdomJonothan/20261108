"""For the dev queries whose frozen top-10 scores zero, find whether a relevant
document exists anywhere in the dense or CE top-100, and at what rank.

CPU only. Reads data/track1_tempo/<d>/qrels_dev.txt, runs/dense_mxbai_<d>_dev.trec,
runs/fusion100_full_ce_s17_20261115_<d>_dev_ce.trec, runs/sub_sft_e2bc_<d>_dev.trec
and results/replacement_v2/candidates_dev.json (for the query list).
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data/track1_tempo"


def read_run(path):
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 4:
            out.setdefault(p[0], []).append((p[2], float(p[3])))
    for q in out:
        out[q].sort(key=lambda x: -x[1])
    return out


def main():
    cand = json.loads((ROOT / "results/replacement_v2/candidates_dev.json").read_text(encoding="utf-8"))
    domains = sorted({r["domain"] for r in cand["queries"]})
    qrels, dense, ce, frozen = {}, {}, {}, {}
    for d in domains:
        g = {}
        for ln in (DATA / d / "qrels_dev.txt").read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) >= 4 and int(p[3]) > 0:
                g.setdefault(p[0], set()).add(p[2])
        qrels[d] = g
        dense[d] = read_run(RUNS / f"dense_mxbai_{d}_dev.trec")
        ce[d] = read_run(RUNS / f"fusion100_full_ce_s17_20261115_{d}_dev_ce.trec")
        frozen[d] = read_run(RUNS / f"sub_sft_e2bc_{d}_dev.trec")

    rows = []
    for r in cand["queries"]:
        d, q = r["domain"], str(r["qid"])
        g = qrels[d].get(q, set())
        top = r["frozen_top10"]
        if any(x in g for x in top):
            continue
        rec = {"domain": d, "qid": q, "n_rel": len(g)}
        for name, run in (("dense", dense[d]), ("ce", ce[d]), ("frozen", frozen[d])):
            ranked = [doc for doc, _ in run.get(q, [])]
            ranks = [i + 1 for i, doc in enumerate(ranked[:100]) if doc in g]
            rec[name] = {"depth": len(ranked[:100]), "best_rank": min(ranks) if ranks else None}
        rows.append(rec)

    def bucket(x):
        if x is None:
            return "none"
        if x <= 20:
            return "<=20"
        if x <= 50:
            return "21-50"
        return "51-100"

    for name in ("dense", "ce", "frozen"):
        counts = {}
        for r in rows:
            counts[bucket(r[name]["best_rank"])] = counts.get(bucket(r[name]["best_rank"]), 0) + 1
        print(name, counts)
    print("zero queries:", len(rows))
    out = {"n_zero": len(rows), "per_query": rows}
    (ROOT / "results/replacement_v2/zero_recall_dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
