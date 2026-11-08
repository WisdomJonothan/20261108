#!/usr/bin/env python3
"""Evaluate temporal re-ranking variants on the frozen candidate pool.

Reads the pair table produced by temporal_discriminator.py (one row per
candidate with ce, s_temp, gap, has_year, label) and re-ranks the candidates.

Variants
  ce                 : cross-encoder score only (two-view baseline)
  zsum(a)            : z(ce) + a * z(s_temp)   (both standardised per query)
  gate(l)            : ce - l * has_year * min(gap,20)/20   (hard time-mismatch penalty)
  gate+temp(l)       : gate(l) with s_temp used as tie-breaker

Reported: macro nDCG@10 over queries, per-query delta vs ce (paired bootstrap),
plus the AUC of s_temp restricted to candidates that carry a year.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    by = {}
    for r in rows:
        by.setdefault(r["qid"], []).append(r)
    return by


def ndcg(ranked, gold, k=10):
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in gold:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    s2 = scores[order]
    i = 0
    while i < len(s2):
        j = i
        while j + 1 < len(s2) and s2[j + 1] == s2[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def variants(rs, alphas, lambdas):
    """Return {name: ranked docid list} for one query."""
    ce = np.asarray([r["ce"] for r in rs], float)
    st = np.asarray([r["s_temp"] for r in rs], float)
    gap = np.asarray([r["gap"] for r in rs], float)
    hy = np.asarray([r["has_year"] for r in rs], float)
    docs = [r["docid"] for r in rs]
    out = {"ce": [docs[i] for i in np.argsort(-ce, kind="mergesort")]}
    for a in alphas:
        s = zscore(ce) + a * zscore(st)
        out[f"zsum{a}"] = [docs[i] for i in np.argsort(-s, kind="mergesort")]
    pen = hy * np.minimum(gap, 20.0) / 20.0
    for l in lambdas:
        s = ce - l * pen
        out[f"gate{l}"] = [docs[i] for i in np.argsort(-s, kind="mergesort")]
        s2 = zscore(ce - l * pen) + 0.5 * zscore(st)
        out[f"gate{l}+t"] = [docs[i] for i in np.argsort(-s2, kind="mergesort")]
    return out, ce, st, hy, docs


def cv_rerank(all_q, cols, folds=5, seed=0):
    """Query-grouped CV: fit logistic on rows of other queries, rank held-out query."""
    from sklearn.linear_model import LogisticRegression
    keys = sorted(all_q)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    fold_of = {q: i % folds for i, q in enumerate(keys)}
    out = {}
    for f in range(folds):
        tr_rows = [r for q in keys if fold_of[q] != f for r in all_q[q]]
        te_keys = [q for q in keys if fold_of[q] == f]
        if not tr_rows:
            continue
        X = np.asarray([[r[c] for c in cols] for r in tr_rows], float)
        y = np.asarray([r["label"] for r in tr_rows], int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        mu, sd = X.mean(0), X.std(0) + 1e-9
        m = LogisticRegression(max_iter=3000, C=1.0).fit((X - mu) / sd, y)
        for q in te_keys:
            rs = all_q[q]
            P = m.predict_proba((np.asarray([[r[c] for c in cols] for r in rs], float) - mu) / sd)[:, 1]
            order = np.argsort(-P, kind="mergesort")
            out.setdefault("cv", {})[q] = [rs[i]["docid"] for i in order]
            ce_order = np.argsort(-np.asarray([r["ce"] for r in rs]), kind="mergesort")
            out.setdefault("ce", {})[q] = [rs[i]["docid"] for i in ce_order]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--alphas", default="0.1,0.25,0.5")
    ap.add_argument("--lambdas", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--out", default="")
    ap.add_argument("--cv", action="store_true")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    lambdas = [float(x) for x in args.lambdas.split(",")]

    all_q = {}
    load_cv = args.cv
    for path in args.pairs:
        dom = os.path.basename(path).replace("temporal_pairs_", "").replace(".jsonl", "")
        for q, rs in load_rows(path).items():
            all_q[f"{dom}:{q}"] = rs

    names = []
    scores = {}
    if load_cv:
        cvout = cv_rerank(all_q, ["ce", "s_temp", "gap", "has_year"])
        for q, ranked in cvout.get("cv", {}).items():
            gold = {r["docid"] for r in all_q[q] if r["label"] == 1}
            if gold:
                scores.setdefault("cv4", {})[q] = ndcg(ranked, gold)
        for q, ranked in cvout.get("ce", {}).items():
            gold = {r["docid"] for r in all_q[q] if r["label"] == 1}
            if gold:
                scores.setdefault("ce", {})[q] = ndcg(ranked, gold)
        names = ["ce", "cv4"]
    auc_time = {"pos": [], "neg": [], "all": []}
    for key, rs in all_q.items():
        gold = {r["docid"] for r in rs if r["label"] == 1}
        if not gold:
            continue
        v, ce, st, hy, docs = variants(rs, alphas, lambdas)
        if not load_cv:
            names = list(v)
        for n, ranked in v.items():
            scores.setdefault(n, {})[key] = ndcg(ranked, gold)
        # s_temp separability restricted to year-bearing candidates
        m = hy == 1
        if m.sum() >= 4 and 0 < sum(r["label"] for r, k in zip(rs, m) if k) < m.sum():
            lb = [r["label"] for r, k in zip(rs, m) if k]
            sc = [r["s_temp"] for r, k in zip(rs, m) if k]
            auc_time["pos"].extend([1] * sum(lb))
            auc_time["neg"].extend([0] * (len(lb) - sum(lb)))
            auc_time["all"].append(auc(sc, lb))

    qs = sorted(set.intersection(*[set(scores[n]) for n in names]))
    rng = np.random.default_rng(0)
    base = np.asarray([scores["ce"][q] for q in qs])
    res = {"n_queries": len(qs), "macro": {}, "delta": {}, "auc_year_subset": {}}
    for n in names:
        arr = np.asarray([scores[n][q] for q in qs])
        d = arr - base
        boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(5000)]
        res["macro"][n] = round(float(arr.mean()), 4)
        res["delta"][n] = {"mean": round(float(d.mean()), 4),
                           "ci": [round(float(np.percentile(boots, 2.5)), 4),
                                  round(float(np.percentile(boots, 97.5)), 4)],
                           "p": round(float(np.mean(np.asarray(boots) > 0)), 3)}
    if load_cv:
        names = ["ce", "cv4"]
    if auc_time["all"]:
        res["auc_year_subset"] = {
            "n_queries": len(auc_time["all"]),
            "mean_per_query": round(float(np.mean(auc_time["all"])), 4),
        }
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
