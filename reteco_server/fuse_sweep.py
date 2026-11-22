"""Sweep the dense:CE fusion weight, chosen on train, reported on dev.

The frozen system uses z-normalised 1:1 fusion, a value inherited from an earlier
experiment rather than tuned. This script fits the single scalar w (CE weight, dense
fixed at 1.0) on the 969 train queries using the leak-free out-of-fold CE runs, then
evaluates exactly that w on dev with the frozen-checkpoint CE rerun.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]
FOLDS = ("f0", "f1", "f2")
WEIGHTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


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


def fuse(dense, ce, w):
    out = {}
    for qid, la in dense.items():
        lb = dict(ce.get(qid, []))
        docs = [d for d, _ in la if d in lb]
        if not docs:
            continue
        da = dict(la)
        sa = np.array([da[d] for d in docs], dtype=float)
        sb = np.array([lb[d] for d in docs], dtype=float)
        for arr in (sa, sb):
            sd = arr.std()
            if sd > 0:
                arr -= arr.mean()
                arr /= sd
        fused = sa + w * sb
        out[qid] = sorted(zip(docs, fused.tolist()), key=lambda x: -x[1])
    return out


def domain_macro(per_query):
    vals = []
    for d in DOMAINS:
        xs = [v for (dom, _), v in per_query.items() if dom == d]
        if xs:
            vals.append(sum(xs) / len(xs))
    return float(np.mean(vals)) if vals else float("nan")


def run_split(split):
    """Return {w: per_query {(domain,qid): ndcg}}."""
    res = {w: {} for w in WEIGHTS}
    for d in DOMAINS:
        dense = read_run(ROOT / "runs" / f"dense_mxbai_{d}_{split}.trec")
        qrels = read_qrels(ROOT / "data" / "track1_tempo" / d / f"qrels_{split}.txt")
        if split == "train":
            per_fold = {}
            for f in FOLDS:
                ce = read_run(ROOT / "runs" / f"{f}_ce_{d}_train_ce.trec")
                per_fold[f] = ce
            for f in FOLDS:
                ce = per_fold[f]
                for w in WEIGHTS:
                    fused = fuse(dense, ce, w)
                    for qid, ranked in fused.items():
                        res[w][(d, qid)] = ndcg10([x for x, _ in ranked[:10]], qrels.get(qid, set()))
        else:
            ce = read_run(ROOT / "runs" / f"frozenrepro_{d}_dev_ce.trec")
            for w in WEIGHTS:
                fused = fuse(dense, ce, w)
                for qid, ranked in fused.items():
                    res[w][(d, qid)] = ndcg10([x for x, _ in ranked[:10]], qrels.get(qid, set()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/replacement_v2/fuse_sweep.json")
    a = ap.parse_args()

    print("fitting on train (fold CE, leak-free) ...")
    train = run_split("train")
    train_macro = {w: domain_macro(v) for w, v in train.items()}
    for w in WEIGHTS:
        print(f"  train w={w:<4} macro {train_macro[w]:.4f}  n={len(train[w])}")
    best_w = max(WEIGHTS, key=lambda w: train_macro[w])
    print(f"chosen on train: w={best_w}")

    print("evaluating on dev (frozen checkpoint, same recipe) ...")
    dev = run_split("dev")
    dev_macro = {w: domain_macro(v) for w, v in dev.items()}
    for w in WEIGHTS:
        print(f"  dev   w={w:<4} macro {dev_macro[w]:.4f}  n={len(dev[w])}")

    out = ROOT / a.out
    out.write_text(json.dumps({
        "weights": list(WEIGHTS),
        "train_macro": {str(w): train_macro[w] for w in WEIGHTS},
        "dev_macro": {str(w): dev_macro[w] for w in WEIGHTS},
        "chosen_w": best_w,
        "dev_gain_vs_w1": dev_macro[best_w] - dev_macro[1.0],
        "per_query_dev": {str(w): {f"{d}|{q}": v for (d, q), v in dev[w].items()} for w in WEIGHTS},
    }, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
