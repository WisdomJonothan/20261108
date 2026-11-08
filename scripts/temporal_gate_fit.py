#!/usr/bin/env python3
"""Fit the temporal gate on the TRAIN split, report dev once (pre-registered).

Protocol (results/HARDNEG_SPEC.md):
  1. fit logistic head on train candidate rows  y = 1[gold]
     features: ce (cross-encoder), s_temp (CE with temporal statement),
               gap (doc-year distance to query interval, clipped), has_year
  2. pick lambda on train by query-grouped CV, maximising train nDCG@10 of
     s = ce + lambda * logit(p_time)
  3. apply the frozen head + lambda to dev candidates, report macro nDCG@10 of
     ce vs gated, per-domain and pooled, with a query bootstrap.

Dev is touched once, only for the final numbers.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(paths):
    by = {}
    for p in paths:
        dom = os.path.basename(p).replace("temporal_pairs_", "").split("_")[0]
        split = "train" if "_train" in p else "dev"
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            by.setdefault((dom, split, r["qid"]), []).append(r)
    return by


def ndcg(ranked, gold, k=10):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


FEATS = ["ce", "s_temp", "gap", "has_year"]


def fit(rows, cols=FEATS):
    from sklearn.linear_model import LogisticRegression
    X = np.asarray([[r[c] for c in cols] for r in rows], float)
    y = np.asarray([r["label"] for r in rows], int)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(max_iter=3000, C=1.0).fit((X - mu) / sd, y)
    return m, mu, sd, cols


def apply_gate(rows, model, mu, sd, cols, lam):
    """s = ce + lam * logit(p_time)."""
    X = (np.asarray([[r[c] for c in cols] for r in rows], float) - mu) / sd
    p = model.predict_proba(X)[:, 1]
    logit = np.log(p / (1 - p + 1e-12) + 1e-12)
    s = np.asarray([r["ce"] for r in rows], float) + lam * logit
    docs = [r["docid"] for r in rows]
    return [docs[i] for i in np.argsort(-s, kind="mergesort")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--lams", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lams = [float(x) for x in args.lams.split(",")]

    tr = load(args.train)
    dv = load(args.dev)

    # ---- step 1: lambda by query-grouped CV inside train ----
    tr_keys = sorted({(d, s, q) for (d, s, q) in tr})
    rng = np.random.default_rng(0)
    rng.shuffle(tr_keys)
    folds = [tr_keys[i::5] for i in range(5)]
    cv_scores = {l: [] for l in lams}
    for f in range(5):
        te = set(folds[f])
        fit_keys = [k for k in tr_keys if k not in te]
        rows_tr = [r for k in fit_keys for r in tr[k]]
        if not rows_tr:
            continue
        m, mu, sd, cols = fit(rows_tr)
        for k in te:
            rows = tr[k]
            gold = {r["docid"] for r in rows if r["label"] == 1}
            if not gold:
                continue
            for l in lams:
                cv_scores[l].append(ndcg(apply_gate(rows, m, mu, sd, cols, l), gold))
    lam_best = max(lams, key=lambda l: float(np.mean(cv_scores[l])) if cv_scores[l] else -1)
    res = {"lam_best": lam_best, "train_cv": {str(l): round(float(np.mean(v)), 4)
                                              for l, v in cv_scores.items() if v}}

    # ---- step 2: refit on all train, apply to dev ----
    m, mu, sd, cols = fit([r for k in tr_keys for r in tr[k]])
    per_q = {"ce": {}, "gated": {}}
    dom_of = {}
    for k in sorted(dv):
        rows = dv[k]
        gold = {r["docid"] for r in rows if r["label"] == 1}
        if not gold:
            continue
        dom = k[0]
        ce_rank = [r["docid"] for r in sorted(rows, key=lambda r: -r["ce"])]
        per_q["ce"][k] = ndcg(ce_rank, gold)
        per_q["gated"][k] = ndcg(apply_gate(rows, m, mu, sd, cols, lam_best), gold)
        dom_of[k] = dom

    def macro(sel):
        vals = [per_q["ce"][k] for k in sel]
        g = [per_q["gated"][k] for k in sel]
        return float(np.mean(vals)), float(np.mean(g))

    keys = sorted(dom_of)
    res["dev"] = {}
    for dom in sorted(set(dom_of.values())):
        sel = [k for k in keys if dom_of[k] == dom]
        be, ga = macro(sel)
        res["dev"][dom] = {"n": len(sel), "ce": round(be, 4), "gated": round(ga, 4),
                           "delta": round(ga - be, 4)}
    be, ga = macro(keys)
    d = np.asarray([per_q["gated"][k] - per_q["ce"][k] for k in keys])
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(5000)]
    res["dev_pooled"] = {"n": len(keys), "ce": round(be, 4), "gated": round(ga, 4),
                         "delta": round(float(d.mean()), 4),
                         "ci": [round(float(np.percentile(boots, 2.5)), 4),
                                round(float(np.percentile(boots, 97.5)), 4)],
                         "p": round(float(np.mean(np.asarray(boots) > 0)), 3)}
    # domain-clustered bootstrap
    doms = sorted(set(dom_of.values()))
    dboot = []
    for _ in range(5000):
        pick = rng.choice(doms, size=len(doms), replace=True)
        vals = [per_q["gated"][k] - per_q["ce"][k] for k in keys if dom_of[k] in pick]
        dboot.append(float(np.mean(vals)))
    res["dev_pooled"]["domain_ci"] = [round(float(np.percentile(dboot, 2.5)), 4),
                                      round(float(np.percentile(dboot, 97.5)), 4)]
    print(json.dumps(res, indent=1))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
