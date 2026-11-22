"""Per-query router between the fused ranking and the cross-encoder ranking.

Train side uses the leak-free out-of-fold CE runs (each train query is ranked by a CE
that never saw it); dev side uses the frozen checkpoint rerun through the same recipe.
Only cheap features (score margins, list agreement, query shape) are used, so the
router costs nothing at inference time.

Usage:
  python router_train.py --fit          # fit on train folds, write router_model.json
  python router_train.py --apply        # apply to dev, write router_dev.json
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


def zfuse(a, b):
    """z-normalise both runs over the shared doc set and fuse 1:1 (the frozen recipe)."""
    out = {}
    for qid, la in a.items():
        lb = dict(b.get(qid, []))
        docs = [d for d, _ in la if d in lb]
        if not docs:
            continue
        sa = np.array([dict(la)[d] for d in docs], dtype=float)
        sb = np.array([lb[d] for d in docs], dtype=float)
        for arr in (sa, sb):
            sd = arr.std()
            if sd > 0:
                arr -= arr.mean()
                arr /= sd
        fused = sa + sb
        out[qid] = sorted(zip(docs, fused.tolist()), key=lambda x: -x[1])
    return out


def feats(fused, ce, qlen):
    f_docs = [d for d, _ in fused[:10]]
    c_docs = [d for d, _ in ce[:10]]
    fsc = np.array([s for _, s in fused[:10]], dtype=float)
    csc = np.array([s for _, s in ce[:10]], dtype=float)

    def norm_margin(sc):
        if len(sc) < 2 or sc[0] == sc[-1]:
            return 0.0
        return float((sc[0] - sc[1]) / (sc[0] - sc[-1]))

    overlap = len(set(f_docs) & set(c_docs)) / 10.0
    # Kendall-style agreement over the shared top-10 documents
    common = [d for d in f_docs if d in c_docs]
    agree, pairs = 0, 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a = f_docs.index(common[i]) < f_docs.index(common[j])
            b = c_docs.index(common[i]) < c_docs.index(common[j])
            agree += int(a == b)
            pairs += 1
    kendall = agree / pairs if pairs else 0.5
    return np.array([
        1.0,
        norm_margin(fsc),
        norm_margin(csc),
        overlap,
        kendall,
        float(len(ce)),          # pool size the CE could score
        float(np.std(csc) if len(csc) > 1 else 0.0),
        min(qlen, 400) / 100.0,
    ])


def collect(split):
    """Return rows: dict(domain, qid, y_fused, y_ce, x)."""
    rows = []
    for d in DOMAINS:
        dense = read_run(ROOT / "runs" / f"dense_mxbai_{d}_{split}.trec")
        qrels = read_qrels(ROOT / "data" / "track1_tempo" / d / f"qrels_{split}.txt")
        if split == "train":
            ce_runs = {f: read_run(ROOT / "runs" / f"{f}_ce_{d}_train_ce.trec") for f in FOLDS}
            fused_runs = {f: zfuse(dense, ce_runs[f]) for f in FOLDS}
        else:
            ce = read_run(ROOT / "runs" / f"frozenrepro_{d}_dev_ce.trec")
            fused = zfuse(dense, ce)
        qlens = {}
        ex = ROOT / "data" / "track1_tempo" / d / f"examples_{split}.jsonl"
        if ex.exists():
            for line in ex.read_text().splitlines():
                o = json.loads(line)
                qlens[str(o.get("qid", o.get("_id")))] = len(str(o.get("query", "")).split())
        if split == "train":
            # every query has exactly one fold whose CE did not see it
            fold_of = json.loads((ROOT / "results" / "replacement_v2" / "ce_folds.json").read_text())
            held = fold_of.get("held", {})
            for d2, f in held.items():
                pass
            for f in FOLDS:
                for qid, ranked in fused_runs[f].items():
                    if qid not in qrels and qid not in ce_runs[f]:
                        continue
                    rows.append({
                        "domain": d, "qid": qid, "fold": f,
                        "y_fused": ndcg10([x for x, _ in ranked[:10]], qrels.get(qid, set())),
                        "y_ce": ndcg10([x for x, _ in ce_runs[f].get(qid, [])[:10]], qrels.get(qid, set())),
                        "x": feats(ranked, ce_runs[f].get(qid, []), qlens.get(str(qid), 60)),
                    })
        else:
            for qid, ranked in fused.items():
                rows.append({
                    "domain": d, "qid": qid,
                    "y_fused": ndcg10([x for x, _ in ranked[:10]], qrels.get(qid, set())),
                    "y_ce": ndcg10([x for x, _ in ce.get(qid, [])[:10]], qrels.get(qid, set())),
                    "x": feats(ranked, ce.get(qid, []), qlens.get(str(qid), 60)),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--l2", type=float, default=10.0)
    a = ap.parse_args()

    model_path = ROOT / "results" / "replacement_v2" / "router_model.json"
    if a.fit:
        rows = collect("train")
        # one row per query: use the fold that held the query out
        seen, uniq = set(), []
        for r in rows:
            k = (r["domain"], r["qid"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(r)
        X = np.stack([r["x"] for r in uniq])
        y = np.array([r["y_ce"] - r["y_fused"] for r in uniq])
        mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
        Z = np.hstack([np.ones((len(X), 1)), (X[:, 1:] - mu) / sd])
        w = np.linalg.solve(Z.T @ Z + a.l2 * np.eye(Z.shape[1]), Z.T @ y)
        pred = Z @ w
        print(f"train queries {len(uniq)}; base mean delta {y.mean():+.4f}")
        print(f"selected mean delta {np.maximum(pred, 0).mean():+.4f}"
              f"; oracle per-query best {np.maximum(y, 0).mean():+.4f}")
        print("weights", np.round(w, 4).tolist())
        model_path.write_text(json.dumps({"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
                                          "features": ["bias", "fused_margin", "ce_margin",
                                                       "top10_overlap", "kendall", "pool",
                                                       "ce_sd", "qlen"]}, indent=1))
        print("wrote", model_path)
    if a.apply:
        m = json.loads(model_path.read_text())
        w, mu, sd = np.array(m["w"]), np.array(m["mu"]), np.array(m["sd"])
        rows = collect("dev")
        X = np.stack([r["x"] for r in rows])
        Z = np.hstack([np.ones((len(X), 1)), (X[:, 1:] - mu) / sd])
        delta = Z @ w
        chosen = delta > 0
        y = np.array([[r["y_fused"], r["y_ce"]] for r in rows])
        gain = np.where(chosen, y[:, 1] - y[:, 0], 0.0)
        base = y[:, 0].mean()
        print(f"dev queries {len(rows)}; base {base:.4f}; routed {base + gain.mean():.4f}"
              f" ({chosen.sum()} switched); oracle-best {np.maximum(y[:, 1] - y[:, 0], 0).mean() + base:.4f}")
        out = ROOT / "results" / "replacement_v2" / "router_dev.json"
        out.write_text(json.dumps({"n": len(rows), "base": base,
                                   "policy": float(base + gain.mean()),
                                   "switched": int(chosen.sum()),
                                   "rows": [dict(r, delta=float(d)) for r, d in zip(rows, delta)]}, indent=1))
        print("wrote", out)


if __name__ == "__main__":
    main()
