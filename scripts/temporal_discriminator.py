#!/usr/bin/env python3
"""Temporal discriminator over frozen dense candidates (Phase 8b).

Hypothesis: a cross-encoder can separate "topically relevant but time-wrong"
documents from relevant ones, using the query's temporal statement (guidance
events + anchors + summary) as a second query view.

For every anchored query we take the frozen dense top-K candidates and score:

  ce      : CE(q_full, d)            (already computed in the two-view run)
  s_temp  : CE(q_temporal, d)        (new; temporal statement as the query)
  diff    : ce - s_temp              (relative time-consistency signal)

plus document-year features derived from the text (gap to the query interval,
whether a year is present, extraction confidence). Labels are qrels (gold=1).

Reported: per-feature within-query AUC, per-query AUC with a query-cluster
bootstrap, query-grouped 5-fold CV AUC for ce vs ce+s_temp, and a placebo where
temporal statements are deranged across queries.
"""
import argparse
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_examples, load_qrels, read_jsonl  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def load_guidance(ddir, split):
    p = os.path.join(ddir, f"guidance_{split}.jsonl")
    out = {}
    if not os.path.isfile(p):
        return out
    for r in read_jsonl(p):
        out[r["id"]] = r.get("query_guidance") or {}
    return out


def years(text):
    return sorted({int(y) for y in YEAR.findall(text or "")})


def interval_of(query, guid):
    parts = [query]
    for k in ("query_summary", "query_temporal_events", "key_time_anchors"):
        v = guid.get(k)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif isinstance(v, dict):
            parts.extend(str(x) for x in v.values())
    ys = years("\n".join(parts))
    if not ys:
        return None
    return (min(ys), max(ys))


def statement_of(guid):
    parts = []
    for k in ("query_temporal_events", "key_time_anchors", "query_summary"):
        v = guid.get(k)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(x) for x in v)
    txt = " ; ".join(p for p in parts if p)
    return txt.strip() or None


def read_run_scores(path, k):
    """TREC 6-col: qid Q0 docid rank score tag."""
    q_scores, q_order = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 6:
                continue
            q, d, rank, s = p[0], p[2], int(p[3]), float(p[4])
            if rank > k:
                continue
            q_scores.setdefault(q, {})[d] = s
            q_order.setdefault(q, []).append((rank, s, d))
    for q in q_order:
        q_order[q].sort()
    return q_scores, q_order


def doc_year_features(text, interval):
    head = (text or "")[:1200]  # title + first ~300 tokens
    ys = years(head)
    if not ys:
        return {"has_year": 0, "gap": 60.0, "n_years": 0, "span": 0}
    lo, hi = interval
    if any(lo - 1 <= y <= hi + 1 for y in ys):
        g = 0.0
    else:
        g = min(abs(y - lo) if y < lo else abs(y - hi) for y in ys)
    return {"has_year": 1, "gap": float(min(g, 60.0)),
            "n_years": len(ys), "span": max(ys) - min(ys)}


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # tie handling
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[pos].sum()
    n_pos, n_neg = pos.sum(), neg.sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dense", required=True)
    ap.add_argument("--ce", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "bge-reranker-v2-m3"))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ddir = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    ids, texts = load_corpus(ddir)
    corpus = dict(zip(ids, texts))
    qids, queries = load_examples(ddir, args.split)
    guid = load_guidance(ddir, args.split)
    pos = {q: set(d.keys()) for q, d in load_qrels(ddir, args.split).items()}

    dense_scores, dense_order = read_run_scores(args.dense, args.k)
    ce_scores, _ = read_run_scores(args.ce, args.k)

    # temporal statements; placebo = statement of a different anchored query
    stmts = {}
    for qi, qid in enumerate(qids):
        iv = interval_of(queries[qi], guid.get(qid, {}))
        st = statement_of(guid.get(qid, {}))
        if iv is not None and st:
            stmts[qid] = (iv, st)
    if args.placebo:
        keys = sorted(stmts)
        shifted = {k: stmts[keys[(i + 1) % len(keys)]][1] for i, k in enumerate(keys)}
        stmts = {k: (v[0], shifted[k]) for k, v in stmts.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=dtype).eval().to(device)

    rows = []
    with torch.inference_mode():
        for qid, (iv, st) in stmts.items():
            cand = [d for _r, _s, d in dense_order.get(qid, [])]
            if not cand:
                continue
            docs = [corpus.get(d, "") for d in cand]
            stoks = tok(st, add_special_tokens=False, truncation=True,
                        max_length=160)["input_ids"]
            st_text = tok.decode(stoks)
            scores = []
            for i in range(0, len(docs), args.batch):
                chunk = docs[i:i + args.batch]
                enc = tok([st_text] * len(chunk), chunk, padding=True,
                          truncation="only_second", max_length=512,
                          return_tensors="pt").to(device)
                s = model(**enc).logits.view(-1).float().cpu().numpy()
                scores.extend(s.tolist())
            for d, s_temp in zip(cand, scores):
                ce = ce_scores.get(qid, {}).get(d)
                if ce is None:
                    continue
                dy = doc_year_features(corpus.get(d, ""), iv)
                rows.append({
                    "qid": qid, "docid": d, "label": int(d in pos.get(qid, set())),
                    "ce": float(ce), "s_temp": float(s_temp),
                    "diff": float(ce - s_temp),
                    "dense": float(dense_scores.get(qid, {}).get(d, 0.0)),
                    **dy,
                })

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- stats ----
    by_q = {}
    for r in rows:
        by_q.setdefault(r["qid"], []).append(r)
    res = {"domain": args.domain, "split": args.split, "placebo": args.placebo,
           "n_queries": len(by_q), "n_rows": len(rows)}

    feats = ["ce", "s_temp", "diff", "dense", "gap", "has_year"]
    res["auc"] = {}
    for f in feats:
        sc, lb = [], []
        for q, rs in by_q.items():
            sc.extend(r[f] for r in rs)
            lb.extend(r["label"] for r in rs)
        a = auc(sc, lb)
        res["auc"][f] = round(a, 4) if a is not None else None

    # per-query AUC for s_temp and diff (queries with both classes)
    per = {f: [] for f in ("ce", "s_temp", "diff")}
    for q, rs in by_q.items():
        lb = [r["label"] for r in rs]
        if sum(lb) == 0 or sum(lb) == len(lb):
            continue
        for f in per:
            a = auc([r[f] for r in rs], lb)
            if a is not None:
                per[f].append(a)
    rng = np.random.default_rng(0)
    res["per_query"] = {}
    for f, vals in per.items():
        arr = np.asarray(vals)
        boots = [rng.choice(arr, size=len(arr), replace=True).mean()
                 for _ in range(2000)] if len(arr) else []
        res["per_query"][f] = {
            "n": len(arr), "mean": round(float(arr.mean()), 4) if len(arr) else None,
            "ci": [round(float(np.percentile(boots, 2.5)), 4),
                   round(float(np.percentile(boots, 97.5)), 4)] if boots else None,
        }

    # query-grouped 5-fold CV over candidate-level logistic model
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        qs = sorted(by_q)
        rng.shuffle(qs)
        folds = [qs[i::5] for i in range(5)]
        res["cv"] = {}
        for name, cols in (("ce", ["ce"]), ("ce+s_temp", ["ce", "s_temp"]),
                           ("ce+diff", ["ce", "diff"]),
                           ("ce+s_temp+gap+has_year", ["ce", "s_temp", "gap", "has_year"])):
            pred = np.zeros(len(rows))
            X = np.asarray([[r[c] for c in cols] for r in rows], dtype=float)
            y = np.asarray([r["label"] for r in rows], dtype=int)
            qidx = np.asarray([r["qid"] for r in rows])
            for f in folds:
                te = np.isin(qidx, f)
                tr = ~te
                if y[tr].sum() == 0 or y[te].sum() == 0 or y[te].sum() == y[te].size:
                    continue
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                m = LogisticRegression(max_iter=2000, C=1.0)
                m.fit((X[tr] - mu) / sd, y[tr])
                pred[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
            mask = pred > 0
            res["cv"][name] = round(float(roc_auc_score(y[mask], pred[mask])), 4)
    except Exception as e:  # pragma: no cover
        res["cv"] = {"error": str(e)}

    with open(args.out.replace(".jsonl", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
