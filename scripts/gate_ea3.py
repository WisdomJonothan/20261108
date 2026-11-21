"""Position-aware expected-gain gate.

Fit on the leak-free fold judge outputs (train queries, 3 folds): expected change
of one swap as a function of the slot's position weight and the judge's margin.
Apply to the 519 dev queries' expanded-action cache (any slot x top-100 candidate).

Model: delta ~ b0 + b1 / slot + b2 * (margin-0.5) + b3 * (margin-0.5) / slot
(ridge, closed form).  Decision: for each query take the judged action with the
largest predicted delta if it is > 0; one swap; leave the ranking otherwise.
"""
import json, os, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else r"D:\RETECO\github_repo"
V2 = os.path.join(ROOT, "results", "replacement_v2")


def load(p):
    with open(p) as f:
        return json.load(f)


def fits():
    X, y = [], []
    for k in ("f0", "f1", "f2"):
        p = os.path.join(V2, "exp_train_%s.json" % k)
        if not os.path.exists(p):
            continue
        pr = load(p)["policies"]["direct"]["pair_records"]
        for r in pr:
            if not r.get("agree"):
                continue
            slot = float(r["slot_rank"])
            m = float(r["merged_challenger"]) - 0.5
            X.append([1.0, 1.0 / slot, m, m / slot])
            y.append(float(r["delta_ndcg"]))
    return np.array(X), np.array(y)


def fit(X, y, lam=1.0):
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def predict(w, slot, margin):
    m = margin - 0.5
    return w @ np.array([1.0, 1.0 / float(slot), m, m / float(slot)])


def main():
    X, y = fits()
    print("train actions", len(y), "helpful", int((y > 1e-9).sum()))
    w = fit(X, y)
    print("weights", np.round(w, 5))
    d = load(os.path.join(V2, "expanded3_prior_dev.json"))
    for k, b in sorted(d["budgets"].items(), key=lambda kv: int(kv[0])):
        total = 0.0
        n_q = 0
        swapped = 0
        gains = 0.0
        per = []
        for q in b["per_query"]:
            acts = [a for a in q.get("actions", []) if a["agree"] and a["mp"] > 0.5]
            best, bestd = None, 0.0
            for a in acts:
                e = predict(w, a["slot"], a["mp"])
                if e > bestd:
                    best, bestd = a, e
            dq = best["delta"] if best is not None else 0.0
            if best is not None:
                swapped += 1
                if dq > 1e-9:
                    gains += dq
            per.append(dict(domain=q["domain"], qid=q["qid"], delta=dq))
            total += dq
            n_q += 1
        # domain macro
        from collections import defaultdict
        by = defaultdict(list)
        for r in per:
            by[r["domain"]].append(r["delta"])
        macro = np.mean([np.mean(v) for v in by.values()])
        print("k=%-3s macro_delta %+.6f swaps %3d gained %+.3f  (baseline 0.358099 -> %.6f)"
              % (k, macro, swapped, gains, 0.35809852326897723 + macro))


if __name__ == "__main__":
    main()
