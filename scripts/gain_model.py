"""Expected-gain selection over the expanded action cache (dev), CPU only.

Fits a logistic "helpful" model and a ridge "delta" model on the leak-free fold
predictions, then replays four selection policies at matched prompt budgets and
reports domain-macro nDCG@10 with a paired bootstrap.

python scripts/gain_model.py --root D:\\RETECO\\github_repo
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]
SEED = 20261120


def fit_logistic(X, y, l2=1.0, iters=60):
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(Z.shape[1])
        w -= np.linalg.solve(H, g)
    return w, mu, sd


def fit_ridge(X, y, l2=1.0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    A = Z.T @ Z + l2 * np.eye(Z.shape[1]); A[0, 0] = 0.0
    w = np.linalg.solve(A, Z.T @ y)
    return w, mu, sd


def apply(model, X):
    w, mu, sd = model
    Z = np.hstack([np.ones((len(X), 1)), (np.asarray(X, float) - mu) / sd])
    return Z @ w


# features available in both the fold pair_records and the dev action cache
FEATS = ["removed_rank", "gap", "judge_margin", "ab_consistency"]


def train(root):
    X, ybin, ydelta = [], [], []
    for f in ("f0", "f1", "f2"):
        p = root / "results/replacement_v2" / ("exp_train_%s.json" % f)
        if not p.exists():
            print("missing", p)
            continue
        d = json.loads(p.read_text())
        for r in d["policies"]["direct"]["pair_records"]:
            X.append([float(r["removed_rank"]),
                      float(r["dense_gap"]) + float(r["ce_gap"]),
                      float(r["judge_margin"]) - 0.5,
                      float(r["ab_order_consistency"])])
            ybin.append(1.0 if r["delta_ndcg"] > 1e-12 else 0.0)
            ydelta.append(float(r["delta_ndcg"]))
    X = np.array(X); ybin = np.array(ybin); ydelta = np.array(ydelta)
    print("train pairs %d, helpful %d" % (len(X), int(ybin.sum())))
    return fit_logistic(X, ybin), fit_ridge(X, ybin * ydelta)


def action_features(a):
    gap = float(a["prior"]) * math.log2(float(a["slot"]) + 2)
    return [float(a["slot"]), gap, float(a["mp"]) - 0.5, 1.0 if a.get("agree") else 0.0]


def macro(per):
    tot = defaultdict(list)
    for r in per:
        tot[r["domain"]].append(r["policy"])
    vals = [sum(v) / len(v) for v in tot.values() if v]
    return sum(vals) / len(vals)


def bootstrap(per, draws=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in per:
        by[r["domain"]].append(r["delta"])
    doms = sorted(by)
    arrs = [np.array(by[d]) for d in doms]
    out = []
    for _ in range(draws):
        s = 0.0
        for a in arrs:
            s += float(a[rng.integers(0, len(a), len(a))].mean())
        out.append(s / len(doms))
    out = np.sort(np.array(out))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def run(root, args):
    m_bin, m_delta = train(root)
    acts = json.loads((root / args.actions).read_text())
    rows = acts["per_query"]
    print("dev queries", len(rows), "actions/query", len(rows[0]["actions"]))

    results = {}
    for policy in ("model", "judge", "cheapgap", "random"):
        for budget in (2, 6, 18):
            per = []
            counts = defaultdict(int)
            for r in rows:
                cand = list(r["actions"])
                if policy == "model":
                    Xa = np.array([action_features(a) for a in cand])
                    p_h = 1 / (1 + np.exp(-apply(m_bin, Xa)))
                    g_h = apply(m_delta, Xa)
                    score = p_h * np.maximum(g_h, 0.0)
                    order = np.argsort(-score)
                elif policy == "judge":
                    order = np.argsort([-a["mp"] for a in cand])
                elif policy == "cheapgap":
                    order = np.argsort([-a["prior"] for a in cand])
                else:
                    rng = random.Random(SEED + int(r["qid"].split("_")[-1]))
                    idx = list(range(len(cand))); rng.shuffle(idx)
                    order = np.array(idx)
                chosen = None
                for i in order[:budget]:
                    a = cand[i]
                    if a["agree"] and a["mp"] > 0.5:
                        chosen = a
                        break
                delta = chosen["delta"] if chosen else 0.0
                if chosen is None:
                    counts["no_swap"] += 1
                elif delta > 1e-12:
                    counts["helpful"] += 1
                elif delta < -1e-12:
                    counts["harmful"] += 1
                else:
                    counts["neutral"] += 1
                per.append({"domain": r["domain"], "qid": r["qid"], "baseline": r["baseline"],
                            "policy": r["baseline"] + delta, "delta": delta})
            key = "%s@%d" % (policy, budget)
            results[key] = {"macro": macro(per), "delta": macro(per) - macro(
                [dict(x, policy=x["baseline"]) for x in per]), "counts": dict(counts),
                "ci95": bootstrap(per)}
            print("%-12s %.4f  %+.4f  %s  %s" % (key, results[key]["macro"],
                                                  results[key]["delta"], results[key]["ci95"],
                                                  dict(counts)))
    dest = root / args.out
    dest.write_text(json.dumps({"features": FEATS, "results": results}, indent=2), encoding="utf-8")
    print("wrote", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\RETECO\github_repo")
    ap.add_argument("--actions", default="results/replacement_v2/expanded3_prior_dev.json")
    ap.add_argument("--out", default="results/replacement_v2/gain_model_dev.json")
    a = ap.parse_args()
    run(Path(a.root), a)


if __name__ == "__main__":
    main()
