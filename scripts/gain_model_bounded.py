"""Expected-gain model for the bounded replacement pool.

Trains on the leak-free fold judgements (969 held-out train queries, 9 pairs each, real
delta_ndcg labels) and applies the fitted model to the dev pairs, choosing at most one
swap per query by predicted gain. Compares against the frozen relevance rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ensemble_rule as E


def feats(rec):
    slot = float(rec["slot_rank"]) + 1.0
    margin = float(rec.get("judge_margin", rec.get("merged_challenger", 0.5))) - 0.5
    dense_gap = float(rec.get("dense_gap", 0.0) or 0.0)
    ce_gap = float(rec.get("ce_gap", 0.0) or 0.0)
    agree = 1.0 if rec.get("ab_order_consistency", rec.get("agree", False)) else 0.0
    return [
        1.0 / slot,
        margin,
        margin / slot,
        dense_gap + ce_gap,
        (dense_gap + ce_gap) / slot,
        agree,
        float(rec.get("incoming_best_rank", 10) or 10) / 20.0,
        1.0 if rec.get("is_dense_source") else 0.0,
    ]


def fit(X, y, l2=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.linalg.solve(Z.T @ Z + l2 * np.eye(Z.shape[1]), Z.T @ y)
    return w, mu, sd


def predict(w, mu, sd, X):
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    return Z @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:/RETECO/github_repo")
    ap.add_argument("--folds", nargs="+", default=[
        "results/replacement_v2/exp_train_f0.json",
        "results/replacement_v2/exp_train_f1.json",
        "results/replacement_v2/exp_train_f2.json",
    ])
    ap.add_argument("--dev", default="results/replacement_v2/exp_direct_probs_dev.json")
    ap.add_argument("--out", default="results/replacement_v2/gain_model_bounded_dev.json")
    args = ap.parse_args()

    root = Path(args.root)
    X, y = [], []
    for rel in args.folds:
        payload = json.loads((root / rel).read_text(encoding="utf-8"))
        for pol in payload["policies"].values():
            for rec in pol.get("pair_records", []):
                X.append(feats(rec))
                y.append(float(rec["delta_ndcg"]))
    X, y = np.array(X), np.array(y)
    w, mu, sd = fit(X, y)
    preds_train = predict(w, mu, sd, X)
    corr = float(np.corrcoef(preds_train, y)[0, 1])
    auc = float(((preds_train[y > 0][:, None] > preds_train[y <= 0][None, :]).mean()) if (y > 0).any() else 0.0)

    dev = E.load_direct(root / args.dev)
    per_query, chosen = [], []
    for key, det in dev.items():
        best = None
        for p in det["pairs"]:
            f = np.array([feats({
                "slot_rank": p["slot_rank"],
                "judge_margin": p.get("merged_challenger", 0.5),
                "dense_gap": (p.get("incoming_dense_score", 0.0) or 0.0) - (p.get("removed_dense_score", 0.0) or 0.0),
                "ce_gap": (p.get("incoming_ce_z", 0.0) or 0.0) - (p.get("removed_ce_z", 0.0) or 0.0),
                "agree": p.get("agree", False),
                "incoming_best_rank": p.get("incoming_best_rank", 10),
                "is_dense_source": p.get("incoming_best_source") == "dense",
            })])
            g = float(predict(w, mu, sd, f)[0])
            if best is None or g > best[0]:
                best = (g, p["delta_ndcg"], p["slot_rank"], p["challenger"])
        delta = best[1] if best and best[0] > 0 else 0.0
        per_query.append({"domain": key[0], "qid": key[1], "baseline": det["baseline"],
                          "policy": det["baseline"] + delta})
        chosen.append({"domain": key[0], "qid": key[1], "pred_gain": best[0] if best else None,
                       "delta": delta, "slot": best[2] if best else None})

    macro = E.domain_macro(per_query)
    base = E.domain_macro([{**r, "policy": r["baseline"]} for r in per_query])
    boot = E.paired_bootstrap(per_query)
    n_sw = sum(1 for c in chosen if abs(c["delta"]) > 1e-12)
    H = sum(1 for c in chosen if c["delta"] > 1e-9)
    Xh = sum(1 for c in chosen if c["delta"] < -1e-9)
    out = {
        "train_pairs": int(len(y)),
        "train_positive": int((y > 0).sum()),
        "train_corr": corr,
        "train_auc": auc,
        "weights": [round(float(v), 5) for v in w],
        "macro_policy": macro,
        "macro_baseline": base,
        "macro_delta": macro - base,
        "bootstrap": boot,
        "swaps": n_sw,
        "helpful": H,
        "harmful": Xh,
        "chosen": chosen,
        "per_query": per_query,
    }
    dest = root / args.out
    dest.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("chosen", "per_query")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
