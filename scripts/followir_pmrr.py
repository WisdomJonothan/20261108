#!/usr/bin/env python3
"""p-MRR (FollowIR) on the RETECO-style FollowIR pseudo-domains.

Implements mteb's `calculate_pmrr`/`rank_score` verbatim and additionally
reports the two directional components (newly relevant docs promoted, newly
irrelevant docs demoted), which are easier to read than a single signed mean.

Usage:
  python scripts/followir_pmrr.py --root ~/RETECO --collection core17:fo --systems dense,ce,frozen
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_trec(path):
    run = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        qid, _, docid, _, score, _ = parts[:6]
        run[qid][docid] = float(score)
    return run


def read_qrels(path):
    q = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 4 and p[1] in ("0", "Q0"):  # TREC qrels: qid 0 docid rel
            q[p[0]][p[2]] = int(p[3])
        elif len(p) >= 3:  # our compact form: qid docid rel
            q[p[0]][p[1]] = int(p[2])
    return q


def rank_of(docs, docid):
    ordered = sorted(docs.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i, (d, _) in enumerate(ordered):
        if d == docid:
            return i + 1
    return len(ordered) + 1


def rank_score(og_rank, new_rank):
    if og_rank >= new_rank:
        return ((1.0 / og_rank) / (1.0 / new_rank)) - 1.0
    return 1.0 - ((1.0 / new_rank) / (1.0 / og_rank))


def pmrr(og_run, chg_run, qog, qchg):
    """mteb semantics: per changed doc, per query, macro over queries."""
    per_q = defaultdict(list)
    promo, demote = defaultdict(list), defaultdict(list)
    for qid in og_run:
        if qid not in chg_run:
            continue
        rel_og = qog.get(qid, {})
        rel_new = qchg.get(qid, {})
        docs = set(rel_og) | set(rel_new)
        for doc in docs:
            a, b = rel_og.get(doc, 0), rel_new.get(doc, 0)
            if a == b:
                continue
            ro = rank_of(og_run[qid], doc)
            rn = rank_of(chg_run[qid], doc)
            per_q[qid].append(rank_score(ro, rn))
            if b > a:  # newly relevant under the changed instruction
                promo[qid].append(1.0 / rn)
            else:  # became irrelevant
                demote[qid].append(1.0 / rn)
    qmeans = [sum(v) / len(v) for v in per_q.values() if v]
    return {
        "p_mrr": sum(qmeans) / len(qmeans) if qmeans else None,
        "n_queries_with_changes": len(qmeans),
        "promotion_mrr": (sum(sum(v) / len(v) for v in promo.values() if v) / max(1, sum(1 for v in promo.values() if v))),
        "demotion_mrr": (sum(sum(v) / len(v) for v in demote.values() if v) / max(1, len(demote))),
        "n_promoted_docs": sum(len(v) for v in promo.values()),
        "n_demoted_docs": sum(len(v) for v in demote.values()),
    }


SYS = {
    "dense": "runs/dense_mxbai_{p}_{c}_dev.trec",
    "ce": "runs/full_ce_{p}_{c}_dev_ce.trec",
    "frozen": "runs/frozen_{p}_{p}_{c}_dev.trec",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--collection", action="append", required=True,
                    help="collection:prefix, e.g. core17:fo (repeatable)")
    ap.add_argument("--systems", default="dense,ce,frozen")
    ap.add_argument("--out", default="results/replacement_v2/followir/pmrr.json")
    a = ap.parse_args()
    root = Path(a.root)
    out = {}
    for spec in a.collection:
        coll, pref = spec.split(":")
        og_run_p = root / SYS["dense"].format(p=pref, c="og")
        _ = og_run_p  # existence checked per system below
        qog = read_qrels(root / f"data/track1_tempo/{pref}_og/qrels_dev.txt")
        qchg = read_qrels(root / f"data/track1_tempo/{pref}_chg/qrels_dev.txt")
        out[coll] = {}
        for sysname in a.systems.split(","):
            ogp = root / SYS[sysname].format(p=pref, c="og")
            chp = root / SYS[sysname].format(p=pref, c="chg")
            if not ogp.exists() or not chp.exists():
                out[coll][sysname] = {"missing": str(ogp.name)}
                continue
            res = pmrr(read_trec(ogp), read_trec(chp), qog, qchg)
            out[coll][sysname] = res
            print(coll, sysname, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                             for k, v in res.items()}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    (Path(a.root) / a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
