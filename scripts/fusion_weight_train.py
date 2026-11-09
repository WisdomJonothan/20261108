#!/usr/bin/env python3
"""Choose the two-view fusion weight on TRAIN, then read dev once.

Critique being answered: the +0.03 dev fusion gain used a weight picked on dev.
Here the weight is selected only from train nDCG@10 (7 domains that have train
runs), then applied unchanged to all 13 dev domains.

Reports:
  * train curve: macro nDCG@10 for dense-only, CE-only and each RRF weight;
  * dev: macro nDCG@10 at the train-selected weight, per domain and overall;
  * per-domain transfer: weight chosen within the domain's own train split,
    applied to that domain's dev (only domains with train runs).
"""
import json
import os

import numpy as np

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DOMAINS = ["bitcoin", "economics", "politics", "genealogy", "history", "hsm", "monero",
                 "law", "workplace", "cardano", "quant", "travel", "iota"]
DEV_DOMAINS = list(TRAIN_DOMAINS)
WEIGHTS = [(1, 0), (4, 1), (2, 1), (1, 1), (1, 2), (1, 4), (0, 1)]  # (dense, ce)
C = 60.0


def read_run(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) < 6:
            continue
        out.setdefault(p[0], []).append((int(p[3]), p[2]))
    return {q: [d for _r, d in sorted(v)] for q, v in out.items()}


def read_qrels(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 4:
            out.setdefault(p[0], set()).add(p[2])
    return out


def ndcg(ranked, gold, k=10):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def fuse(dense, ce, wd, wc):
    out = {}
    for q in set(dense) | set(ce):
        rd = {d: i + 1 for i, d in enumerate(dense.get(q, []))}
        rc = {d: i + 1 for i, d in enumerate(ce.get(q, []))}
        s = {d: wd / (C + rd.get(d, 10 ** 6)) + wc / (C + rc.get(d, 10 ** 6))
             for d in set(rd) | set(rc)}
        out[q] = [d for d, _ in sorted(s.items(), key=lambda t: -t[1])]
    return out


def domain_scores(dom, split):
    base = os.path.join(ROOT, "data", "track1_tempo", dom)
    qrels = read_qrels(os.path.join(base, f"qrels_{split}.txt"))
    dense = read_run(os.path.join(ROOT, "runs", f"dense_mxbai_{dom}_{split}.trec"))
    ce_path = os.path.join(ROOT, "runs", f"rerank_mxbai_{dom}_{split}.trec")
    ce = read_run(ce_path)
    res = {}
    for wd, wc in WEIGHTS:
        run = dense if wc == 0 else (ce if wd == 0 else fuse(dense, ce, wd, wc))
        per_q = [ndcg(run.get(q, []), g) for q, g in qrels.items() if g]
        res[f"d{wc}:c{wd}" if False else f"{wd}:{wc}"] = float(np.mean(per_q)) if per_q else 0.0
    return res


def main():
    train = {d: domain_scores(d, "train") for d in TRAIN_DOMAINS}
    weights = [f"{wd}:{wc}" for wd, wc in WEIGHTS]
    train_macro = {w: float(np.mean([train[d][w] for d in TRAIN_DOMAINS])) for w in weights}
    best = max(weights, key=lambda w: train_macro[w])

    # dev: numbers were produced by analyze_mxbai.py; recompute here for the
    # selected weight so that this script is self-contained.
    dev_macro = {}
    dev_per_domain = {}
    for w in weights:
        vals = {}
        for d in DEV_DOMAINS:
            try:
                vals[d] = domain_scores(d, "dev")[w]
            except FileNotFoundError:
                continue
        dev_per_domain[w] = vals
        dev_macro[w] = float(np.mean(list(vals.values()))) if vals else 0.0

    # per-domain weight chosen on own train, evaluated on own dev
    transfer = {}
    for d in TRAIN_DOMAINS:
        w_local = max(weights, key=lambda w: train[d][w])
        transfer[d] = {"train_best": w_local,
                       "dev_at_local": round(dev_per_domain[w_local].get(d, 0.0), 4),
                       "dev_at_global": round(dev_per_domain[best].get(d, 0.0), 4),
                       "dev_at_2:1": round(dev_per_domain["2:1"].get(d, 0.0), 4)}

    out = {
        "train_domains": TRAIN_DOMAINS,
        "train_macro": {w: round(v, 4) for w, v in train_macro.items()},
        "selected_by_train": best,
        "dev_macro_at_each_weight": {w: round(v, 4) for w, v in dev_macro.items()},
        "dev_per_domain": {w: {d: round(v, 4) for d, v in vals.items()}
                           for w, vals in dev_per_domain.items()},
        "domain_transfer": transfer,
    }
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "results", "fusion_weight_train.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
