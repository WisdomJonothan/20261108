"""Pick the queries where a second opinion is worth paying for.

A query is "uncertain" for the Direct arm when it either made no swap or its
chosen pair had a merged challenger probability below ``--margin``.  Those are
the queries a cascade would send to a second, condition-focused judge.
"""
import argparse, json, sys
from pathlib import Path

OUT = Path("results/replacement_v2/cascade_subset.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.9)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    exp = Path(a.root) / "results/replacement_v2/exp_direct_context_dev.json"
    d = json.load(open(exp, encoding="utf-8"))
    recs = d["policies"]["direct"]["pair_records"]
    chosen = {}
    for r in recs:
        if r.get("chosen"):
            chosen[(r["domain"], r["qid"])] = r["merged_challenger"]
    queries = sorted({(r["domain"], r["qid"]) for r in recs})
    uncertain = [q for q in queries if chosen.get(q, 0.0) < a.margin]
    payload = {"margin": a.margin, "n_queries": len(queries), "n_selected": len(uncertain),
               "selected": [{"domain": d_, "qid": q} for d_, q in uncertain],
               "with_swap": sum(1 for q in uncertain if q in chosen),
               "without_swap": sum(1 for q in uncertain if q not in chosen)}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print("selected %d of %d (with swap %d, without %d)"
          % (payload["n_selected"], payload["n_queries"], payload["with_swap"],
             payload["without_swap"]))


if __name__ == "__main__":
    main()
