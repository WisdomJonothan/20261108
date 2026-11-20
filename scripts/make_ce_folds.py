"""Build leakage-free CE training folds for the gating study (step 5).

Frozen protocol (replacement_study_20261120.json):
  * 1211 official train queries -> 969 gate-train / 242 calibration (train_split.json)
  * within the 969, three stratified folds; each fold model is trained on the other two
  * a fourth model trained on all 969 supplies rankings for the 242 calibration queries
  * every query is ranked only by a model that never saw it

Outputs (all under ~/RETECO):
  results/ce_folds/f{0,1,2}/g_f{k}_<domain>.jsonl   training groups for fold k
  results/ce_folds/f{0,1,2}/ex_f{k}_<domain>.jsonl  held-out examples for fold k
  results/ce_folds/fall/g_all_<domain>.jsonl        all 969 training groups
  results/ce_folds/fall/ex_all_<domain>.jsonl       calibration examples
  results/replacement_v2/ce_folds.json              assignment + provenance

The group jsonl rows are copied verbatim (same 20-doc supervision as the frozen CE);
only the query subset changes.
"""
import argparse
import hashlib
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]


def load_split(path):
    split = json.load(open(path, encoding="utf-8"))
    dom = split["domains"]
    out = {}
    for d in DOMAINS:
        v = dom[d]
        if isinstance(v, dict):
            tr = v.get("train") or v.get("n_train") or v.get("gate_train")
            ca = v.get("calibration") or v.get("calib") or v.get("n_calibration")
        else:  # list of [train, calib]
            tr, ca = v[0], v[1]
        out[d] = ([str(q) for q in tr], [str(q) for q in ca])
    return split, out


def assign_folds(train_by_domain, seed):
    fold_of = {}
    for d in DOMAINS:
        qids = sorted(train_by_domain[d])
        rng = random.Random("%s|%s" % (seed, d))
        rng.shuffle(qids)
        for i, q in enumerate(qids):
            fold_of[q] = i % 3
    return fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--groups-prefix", default="results/ce_train_groups_bf")
    ap.add_argument("--split", default="results/replacement_v2/train_split.json")
    ap.add_argument("--outdir", default="results/ce_folds")
    ap.add_argument("--manifest", default="results/replacement_v2/ce_folds.json")
    ap.add_argument("--seed", default="20261120")
    args = ap.parse_args()
    R = args.root

    split, by_dom = load_split(os.path.join(R, args.split))
    fold_of = assign_folds({d: by_dom[d][0] for d in DOMAINS}, args.seed)
    calib = {d: by_dom[d][1] for d in DOMAINS}
    n_train = sum(len(by_dom[d][0]) for d in DOMAINS)
    n_calib = sum(len(v) for v in calib.values())

    # held-out query sets: fold k -> queries with fold_of == k
    held = {k: {d: [] for d in DOMAINS} for k in range(3)}
    for d in DOMAINS:
        for q in by_dom[d][0]:
            held[fold_of[q]][d].append(q)

    examples_cache = {}
    for d in DOMAINS:
        rows = {}
        with open(os.path.join(R, "data", "track1_tempo", d, "examples_train.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[str(r["id"])] = line
        examples_cache[d] = rows

    manifest = {"seed": args.seed, "n_train": n_train, "n_calib": n_calib,
                "groups_prefix": args.groups_prefix, "folds": {}, "calibration": calib}
    stats = {}
    for k in range(3):
        d_out = os.path.join(R, args.outdir, "f%d" % k)
        os.makedirs(d_out, exist_ok=True)
        keep = {q for q, f in fold_of.items() if f != k}
        heldq = {q for q, f in fold_of.items() if f == k}
        ngroups = 0
        for d in DOMAINS:
            src = os.path.join(R, "%s_%s.jsonl" % (args.groups_prefix, d))
            gp = os.path.join(d_out, "g_f%d_%s.jsonl" % (k, d))
            with open(src, encoding="utf-8") as f, open(gp, "w", encoding="utf-8") as g:
                for line in f:
                    qid = str(json.loads(line)["qid"])
                    if qid in keep:
                        g.write(line)
                        ngroups += 1
            ep = os.path.join(d_out, "ex_f%d_%s.jsonl" % (k, d))
            with open(ep, "w", encoding="utf-8") as g:
                for q in held[k][d]:
                    if q in examples_cache[d]:
                        g.write(examples_cache[d][q])
        stats["fold%d" % k] = {"train_queries": len(keep), "held_queries": len(heldq),
                               "train_groups": ngroups}
        manifest["folds"]["f%d" % k] = {
            "held": {d: sorted(held[k][d]) for d in DOMAINS},
            "train_queries": sorted(keep)}

    # all-969 model for the calibration queries
    d_out = os.path.join(R, args.outdir, "fall")
    os.makedirs(d_out, exist_ok=True)
    allq = {q for q in fold_of}
    ngroups = 0
    for d in DOMAINS:
        src = os.path.join(R, "%s_%s.jsonl" % (args.groups_prefix, d))
        gp = os.path.join(d_out, "g_all_%s.jsonl" % d)
        with open(src, encoding="utf-8") as f, open(gp, "w", encoding="utf-8") as g:
            for line in f:
                if str(json.loads(line)["qid"]) in allq:
                    g.write(line)
                    ngroups += 1
        ep = os.path.join(d_out, "ex_all_%s.jsonl" % d)
        with open(ep, "w", encoding="utf-8") as g:
            for q in calib[d]:
                if q in examples_cache[d]:
                    g.write(examples_cache[d][q])
    stats["fall"] = {"train_queries": len(allq), "held_queries": n_calib,
                     "train_groups": ngroups}
    manifest["stats"] = stats
    manifest["fold_of"] = fold_of

    os.makedirs(os.path.dirname(os.path.join(R, args.manifest)), exist_ok=True)
    with open(os.path.join(R, args.manifest), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    print(json.dumps({"stats": stats, "n_fold_of": len(fold_of)}, indent=2))
    print("wrote", os.path.join(R, args.manifest))


if __name__ == "__main__":
    main()
