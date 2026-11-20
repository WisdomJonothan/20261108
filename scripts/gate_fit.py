#!/usr/bin/env python3
"""Step 5: fit and evaluate the swap gate.

Protocol (frozen in configs/replacement_study_20261120.json):
  * fit on the 969 gate-train queries, whose judge features come from the three
    fold models (every query is scored by a CE that never saw it)
  * pick the operating threshold on the 242 calibration queries only
  * evaluate on the 519 official dev queries of the frozen 0.3581 system

Records are the Direct-arm pair records of an exp_*.json.  Each query contributes
at most one swap: among the pairs where both A/B orders agree on the challenger
the harness takes the largest merged probability, and the gate may veto it.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]
FEATURES_JUDGE = ["removed_rank", "incoming_best_rank", "is_dense_source", "dense_gap",
                  "ce_gap", "rank_disagreement", "judge_margin", "ab_order_consistency"]
FEATURES_CHEAP = ["removed_rank", "incoming_best_rank", "is_dense_source", "dense_gap",
                  "ce_gap", "rank_disagreement"]


def read_qrels(path):
    gold = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                gold.setdefault(p[0], set()).add(p[2])
    return gold


def ndcg10(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def load_side(root, cand_path, exp_path, qrels_split, policy="direct"):
    cand = json.loads((root / cand_path).read_text(encoding="utf-8"))
    exp = json.loads((root / exp_path).read_text(encoding="utf-8"))
    tops, domains = {}, {}
    for row in cand["queries"]:
        tops[row["qid"]] = row["frozen_top10"]
        domains[row["qid"]] = row["domain"]
    qrels = {}
    for d in DOMAINS:
        qrels.update(read_qrels(root / "data/track1_tempo" / d / f"qrels_{qrels_split}.txt"))
    base = {q: ndcg10(tops[q], qrels.get(q, set())) for q in tops}
    recs = []
    for r in exp["policies"][policy]["pair_records"]:
        fwd = 0.5  # merged already averages both orders
        margin = float(r["merged_challenger"])
        recs.append({
            "domain": r["domain"], "qid": r["qid"], "slot": r["slot_rank"],
            "removed_rank": r["slot_rank"],
            "incoming_best_rank": r["incoming_best_rank"],
            "is_dense_source": int(r["incoming_best_source"] == "dense"),
            "dense_gap": float(r.get("incoming_dense_score") or 0.0)
            - float(r.get("removed_dense_score") or 0.0),
            "ce_gap": float(r.get("incoming_ce_z") or 0.0)
            - float(r.get("removed_ce_z") or 0.0),
            "rank_disagreement": int(abs(r["slot_rank"] - r["incoming_best_rank"])),
            "judge_margin": margin,
            "ab_order_consistency": int(bool(r.get("agree"))),
            "eligible": int(bool(r.get("agree")) and margin > 0.5),
            "delta": float(r["delta_ndcg"]),
            "chosen": int(bool(r.get("chosen"))),
        })
    return {"base": base, "domains": domains, "recs": recs}


def fit_lr(X, y, l2=1.0, iters=60):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y) + l2 * np.r_[0.0, w[1:]]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.diag(np.r_[0.0, np.ones(Xb.shape[1] - 1)])
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def predict(w, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1.0 / (1.0 + np.exp(-(Xb @ w)))


def decision_matrix(side, feats, w=None, tau=None, fixed_margin=None, gap_rule=False,
                    cheap_rule=False):
    """Per-query policy nDCG for a decision rule, plus swap stats and judge-call count."""
    by_q = {}
    for r in side["recs"]:
        by_q.setdefault((r["domain"], r["qid"]), []).append(r)
    per_q = {}
    stats = {"helpful": 0, "harmful": 0, "neutral": 0, "no_swap": 0,
             "gained": 0.0, "lost": 0.0, "judge_calls": 0, "pairs": 0}
    for (d, q), recs in by_q.items():
        stats["pairs"] += len(recs)
        elig = [r for r in recs if r["eligible"]]
        if not elig:
            per_q[q] = 0.0
            stats["no_swap"] += 1
            continue
        best = max(elig, key=lambda r: r["judge_margin"])
        accept = False
        if gap_rule:
            accept = best["ce_gap"] > 0.0
            stats["judge_calls"] += len(recs)
        elif cheap_rule:
            accept = best["ce_gap"] > 0.0 or best["dense_gap"] > 0.0
            stats["judge_calls"] += len(recs)
        elif fixed_margin is not None:
            accept = best["judge_margin"] > fixed_margin
            stats["judge_calls"] += len(recs)
        else:
            v = float(predict(w, np.array([[r[f] for f in feats] for r in [best]]))[0])
            accept = v > tau
            stats["judge_calls"] += len(recs)
        if accept:
            per_q[q] = best["delta"]
            if best["delta"] > 1e-12:
                stats["helpful"] += 1
                stats["gained"] += best["delta"]
            elif best["delta"] < -1e-12:
                stats["harmful"] += 1
                stats["lost"] += best["delta"]
            else:
                stats["neutral"] += 1
        else:
            per_q[q] = 0.0
            stats["no_swap"] += 1
    return per_q, stats


def macro(side, per_q):
    vals = []
    for d in DOMAINS:
        qs = [q for q, dd in side["domains"].items() if dd == d and q in per_q]
        base = np.array([side["base"][q] for q in qs])
        pol = np.array([side["base"][q] + per_q[q] for q in qs])
        vals.append((pol.mean() - base.mean()) if len(qs) else 0.0)
    return float(np.mean(vals))


def boot(side, per_q, draws=10000, seed=20261120):
    rng = np.random.default_rng(seed)
    arrs = {}
    for d in DOMAINS:
        qs = [q for q, dd in side["domains"].items() if dd == d and q in per_q]
        if qs:
            arrs[d] = np.array([per_q[q] for q in qs])
    sims = np.empty(draws)
    for i in range(draws):
        sims[i] = np.mean([v[rng.integers(len(v), size=len(v))].mean() for v in arrs.values()])
    return {"delta": float(np.mean([v.mean() for v in arrs.values()])),
            "ci95": [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))],
            "p_le_0": float((sims <= 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--train", nargs="+", required=True,
                    help="pairs of candidates_train_f<K>.json:exp_train_f<K>.json")
    ap.add_argument("--calib", required=True, help="candidates_train_fall.json:exp_train_fall.json")
    ap.add_argument("--dev", default="results/replacement_v2/candidates_dev.json:"
                                    "results/replacement_v2/exp_direct_context_dev.json")
    ap.add_argument("--out", default="results/replacement_v2/gate_eval.json")
    args = ap.parse_args()
    root = Path(args.root)

    def load(spec, split):
        c, e = spec.split(":")
        return load_side(root, c, e, split)

    tr = [load(s, "train") for s in args.train]
    ca = load(args.calib, "train")
    dv = load(args.dev, "dev")

    Xtr = np.array([[r[f] for f in FEATURES_JUDGE] for s in tr for r in s["recs"]], dtype=float)
    ytr = np.array([1.0 if r["delta"] > 1e-12 else 0.0 for s in tr for r in s["recs"]])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    w = fit_lr((Xtr - mu) / sd, ytr)
    Xc = np.array([[r[f] for f in FEATURES_CHEAP] for s in tr for r in s["recs"]], dtype=float)
    muc, sdc = Xc.mean(0), Xc.std(0)
    sdc[sdc == 0] = 1.0
    wc = fit_lr((Xc - muc) / sdc, ytr)

    def score(w_, feats, mu_, sd_, side, recs):
        Z = (np.array([[r[f] for f in feats] for r in recs], dtype=float) - mu_) / sd_
        return predict(w_, Z)

    # threshold on calibration only
    elig_scores = {}
    for side_name, side in (("calib", ca), ("dev", dv)):
        G = score(w, FEATURES_JUDGE, mu, sd, side, side["recs"])
        C = score(wc, FEATURES_CHEAP, muc, sdc, side, side["recs"])
        for r, g, c in zip(side["recs"], G, C):
            r["gate_score"] = float(g)
            r["cheap_score"] = float(c)
    grid = np.quantile([r["gate_score"] for r in ca["recs"] if r["eligible"]], np.linspace(0, 1, 21))
    best = None
    for tau in sorted(set(float(x) for x in grid)):
        pq, st = decision_matrix(ca, FEATURES_JUDGE, w, tau)
        m = macro(ca, pq)
        if best is None or m > best[1]:
            best = (tau, m, st)
    tau = best[0]

    out = {"features_judge": FEATURES_JUDGE, "weights_judge": w.tolist(),
           "mean": mu.tolist(), "sd": sd.tolist(),
           "features_cheap": FEATURES_CHEAP, "weights_cheap": wc.tolist(),
           "mean_cheap": muc.tolist(), "sd_cheap": sdc.tolist(),
           "calibration": {"tau": tau, "macro_delta": best[1], "stats": best[2],
                           "n_pairs": len(ca["recs"]),
                           "base_beneficial": int(sum(r["delta"] > 1e-12 for r in ca["recs"]))},
           "rules": {}}
    # calibration-side rule comparison
    for name, kw in (("direct_all", dict(tau=0.0)), ("fixed_margin_0.7", dict(fixed_margin=0.7)),
                     ("fixed_margin_0.9", dict(fixed_margin=0.9)), ("ce_gap_rule", dict(gap_rule=True)),
                     ("ce_or_dense_gap", dict(cheap_rule=True))):
        pq, st = decision_matrix(ca, FEATURES_JUDGE, w, **kw)
        out["rules"][name] = {"calib_macro_delta": macro(ca, pq), "stats": st}

    def eval_dev(per_q, st):
        b = boot(dv, per_q)
        return {"macro_delta": macro(dv, per_q), "bootstrap": b, "stats": st,
                "points": {"gained": st["gained"], "lost": st["lost"]}}

    pq, st = decision_matrix(dv, FEATURES_JUDGE, w, tau)
    out["dev"] = {"gate_judge": eval_dev(pq, st)}
    pq, st = decision_matrix(dv, FEATURES_JUDGE, w, 0.0)
    out["dev"]["direct_all"] = eval_dev(pq, st)
    for name, kw in (("fixed_margin_0.7", dict(fixed_margin=0.7)),
                     ("fixed_margin_0.9", dict(fixed_margin=0.9)),
                     ("ce_gap_rule", dict(gap_rule=True)),
                     ("ce_or_dense_gap", dict(cheap_rule=True))):
        pq, st = decision_matrix(dv, FEATURES_JUDGE, w, **kw)
        out["dev"][name] = eval_dev(pq, st)
    # pre-call cheap gate: only judge pairs whose cheap score clears its own tau (chosen on calib)
    cheap_grid = np.quantile([r["cheap_score"] for r in ca["recs"]], np.linspace(0, 1, 21))
    bestc = None
    for t in sorted(set(float(x) for x in cheap_grid)):
        byq = {}
        for r in ca["recs"]:
            byq.setdefault((r["domain"], r["qid"]), []).append(r)
        per_q, keep_pairs, tot = {}, 0, 0
        for (d, q), recs in byq.items():
            tot += len(recs)
            elig = [r for r in recs if r["eligible"] and r["cheap_score"] > t]
            keep_pairs += sum(1 for r in recs if r["cheap_score"] > t)
            if not elig:
                per_q[q] = 0.0
            else:
                b = max(elig, key=lambda r: r["judge_margin"])
                per_q[q] = b["delta"]
        m = macro(ca, per_q)
        if bestc is None or m > bestc[1]:
            bestc = (t, m, keep_pairs / max(tot, 1))
    out["calibration"]["cheap_tau"] = bestc[0]
    out["calibration"]["cheap_call_fraction"] = bestc[2]
    byq = {}
    for r in dv["recs"]:
        byq.setdefault((r["domain"], r["qid"]), []).append(r)
    per_q, keep, tot = {}, 0, 0
    st = {"helpful": 0, "harmful": 0, "neutral": 0, "no_swap": 0, "gained": 0.0, "lost": 0.0}
    for (d, q), recs in byq.items():
        tot += len(recs)
        keep += sum(1 for r in recs if r["cheap_score"] > bestc[0])
        elig = [r for r in recs if r["eligible"] and r["cheap_score"] > bestc[0]]
        if not elig:
            per_q[q] = 0.0
            st["no_swap"] += 1
            continue
        b = max(elig, key=lambda r: r["judge_margin"])
        per_q[q] = b["delta"]
        if b["delta"] > 1e-12:
            st["helpful"] += 1
            st["gained"] += b["delta"]
        elif b["delta"] < -1e-12:
            st["harmful"] += 1
            st["lost"] += b["delta"]
        else:
            st["neutral"] += 1
    out["dev"]["precall_cheap_gate"] = {"macro_delta": macro(dv, per_q),
                                        "bootstrap": boot(dv, per_q), "stats": st,
                                        "judge_call_fraction": keep / max(tot, 1),
                                        "points": {"gained": st["gained"], "lost": st["lost"]}}
    out["dev"]["baseline_macro"] = float(np.mean([np.mean([dv["base"][q] for q, dd in dv["domains"].items()
                                                           if dd == d]) for d in DOMAINS]))
    (root / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else
                               {x: y for x, y in vv.items() if x in ("macro_delta", "ci95", "p_le_0", "judge_call_fraction")})
                          for kk, vv in v.items()} if isinstance(v, dict) else v
                      for k, v in out["dev"].items()}, indent=1))
    print("tau", tau, "wrote", root / args.out)


if __name__ == "__main__":
    main()
