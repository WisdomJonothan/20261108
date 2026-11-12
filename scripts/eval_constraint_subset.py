#!/usr/bin/env python
"""Split a dev run by whether the query loses a constraint under title-only
input, and report domain-macro nDCG@10 on each side.

The split comes from `results/QUERY_CONSTRAINT_FLAGS.json`
(`any_lost`: a cue appears in the body but not in the title). This is the
mechanism test for the constraint-faithful query representation: if keeping the
cue-bearing sentences helps, the gain must concentrate on the `any_lost=True`
subset.

Usage:
  eval_constraint_subset.py --run-glob "results/sub_sft_e2bc_*_dev.trec" \
      --data data/track1_tempo --split dev [--json out.json]
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "official"))
sys.path.insert(0, HERE)
from ir_metrics import ndcg_at_k  # noqa: E402
from lib_reteco import load_qrels, read_trec_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-glob", required=True)
    ap.add_argument("--data", default="data/track1_tempo")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--flags", default="results/QUERY_CONSTRAINT_FLAGS.json")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    flags = json.load(open(args.flags, encoding="utf-8"))
    per_domain = {}
    for path in sorted(glob.glob(args.run_glob)):
        base = os.path.basename(path)
        domain = None
        for d in sorted(os.listdir(args.data)):
            if ("_%s_%s.trec" % (d, args.split)) in base:
                domain = d
                break
        if domain is None:
            continue
        qrels = load_qrels(os.path.join(args.data, domain), args.split)
        run = read_trec_run(path)
        acc = {"lost": [], "kept": []}
        for q, gold in qrels.items():
            if q not in run:
                continue
            f = flags.get(str(q))
            if f is None:
                continue
            ranked = [d for _r, _s, d in run[q]]
            nd = ndcg_at_k(ranked, set(gold), 10)
            acc["lost" if f["any_lost"] else "kept"].append(nd)
        per_domain[domain] = {
            "n_lost": len(acc["lost"]), "n_kept": len(acc["kept"]),
            "lost": sum(acc["lost"]) / len(acc["lost"]) if acc["lost"] else None,
            "kept": sum(acc["kept"]) / len(acc["kept"]) if acc["kept"] else None,
        }

    def macro(key):
        vals = [v[key] for v in per_domain.values() if v[key] is not None]
        return sum(vals) / len(vals) if vals else None

    out = {"run_glob": args.run_glob, "split": args.split,
           "macro_lost": macro("lost"), "macro_kept": macro("kept"),
           "n_lost": sum(v["n_lost"] for v in per_domain.values()),
           "n_kept": sum(v["n_kept"] for v in per_domain.values()),
           "per_domain": per_domain}
    print("%-14s %5s %5s %8s %8s" % ("domain", "lost", "kept", "nDCG-lost", "nDCG-kept"))
    for d, v in per_domain.items():
        print("%-14s %5d %5d %8s %8s" % (
            d, v["n_lost"], v["n_kept"],
            "-" if v["lost"] is None else "%.4f" % v["lost"],
            "-" if v["kept"] is None else "%.4f" % v["kept"]))
    print("macro over domains: lost %.4f (n=%d) | kept %.4f (n=%d)" % (
        out["macro_lost"], out["n_lost"], out["macro_kept"], out["n_kept"]))
    if args.json:
        json.dump(out, open(args.json, "w", encoding="utf-8"), indent=1)
        print("written ->", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
