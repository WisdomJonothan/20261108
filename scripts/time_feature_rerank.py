"""Temporal-feature re-ranking ablation on frozen dense candidates.

Design (fixed before running, no dev tuning):
  * candidates = dense top-100 (frozen run files);
  * features per (query, document) are *time* signals plus controls;
  * a logistic regression is fitted on the TRAIN split (labels from qrels_train)
    and applied to DEV; the dev number is reported once;
  * ablations: dense-only, +time, +controls-only, time-only.

Time features are derived only from the query text/guidance and the document
text (the corpus has no timestamp metadata):
  * years mentioned in the document (count, presence);
  * overlap between query years and document years, and their distance;
  * lexical match of the query's official time anchors (guidance
    key_time_anchors / query_temporal_signals) inside the document;
  * the guidance `is_temporal_query` flag.

Controls (to show the model is not just learning length/lexical overlap):
  * document token length, query token length, document-query lexical overlap.

No test labels exist; qrels are read only for fitting on train and scoring.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import (load_corpus, load_examples, load_qrels,  # noqa: E402
                        read_trec_run, write_trec_run)

ROOT = r"D:\RETECO"
YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
TOKEN = re.compile(r"[A-Za-z0-9]+")


def years_in(text):
    return [int(y) for y in YEAR.findall(text or "")]


def tokenize(text):
    return TOKEN.findall((text or "").lower())


def load_guidance(domain, split):
    path = os.path.join(ROOT, "data", "track1_tempo", domain, f"guidance_{split}.jsonl")
    out = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = r["query_guidance"]
    return out


def build(domain, split, topk=100):
    data = os.path.join(ROOT, "data", "track1_tempo", domain)
    ids, texts = load_corpus(data)
    corpus = dict(zip(ids, texts))
    qids, queries = load_examples(data, split)
    guidance = load_guidance(domain, split)
    run = read_trec_run(os.path.join(ROOT, "runs", f"dense_bge_{domain}_{split}.trec"))

    rows = []
    for qid, qtext in zip(qids, queries):
        docs = [t[2] for t in run[qid][:topk]]
        qtok = set(tokenize(qtext))
        q_years = set(years_in(qtext))
        g = guidance.get(qid, {})
        anchors = " ".join(g.get("key_time_anchors", []) +
                           g.get("query_temporal_signals", []))
        anchor_tokens = set(tokenize(anchors)) - {"the", "a", "of", "and", "in", "to"}
        is_temporal = 1.0 if g.get("is_temporal_query") else 0.0
        for d in docs:
            dtext = corpus.get(d, "")
            dtok = tokenize(dtext)
            dset = set(dtok)
            d_years = set(years_in(dtext))
            overlap_years = len(q_years & d_years)
            if q_years and d_years:
                dist = min(abs(a - b) for a in q_years for b in d_years)
                dist = min(dist, 60) / 60.0
            else:
                dist = 1.0
            rows.append({
                "qid": qid, "doc": d,
                "dense_score": float([t[1] for t in run[qid][:topk] if t[2] == d][0]),
                "doc_has_year": 1.0 if d_years else 0.0,
                "doc_year_count": float(min(len(d_years), 5)) / 5.0,
                "query_has_year": 1.0 if q_years else 0.0,
                "year_overlap": float(min(overlap_years, 3)) / 3.0,
                "year_distance": dist,
                "anchor_match": (len(anchor_tokens & dset) / max(1, len(anchor_tokens)))
                if anchor_tokens else 0.0,
                "is_temporal": is_temporal,
                # controls
                "doc_len": np.log1p(len(dtok)),
                "query_len": np.log1p(len(qtok)),
                "lex_overlap": len(qtok & dset) / max(1, len(qtok)),
            })
    return rows


TIME_FEATS = ["doc_has_year", "doc_year_count", "query_has_year", "year_overlap",
              "year_distance", "anchor_match", "is_temporal"]
CTRL_FEATS = ["doc_len", "query_len", "lex_overlap"]
DENSE_FEATS = ["dense_score"]


def fit_predict(train_rows, test_rows, feats):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr = np.array([[r[f] for f in feats] for r in train_rows])
    ytr = np.array([r["label"] for r in train_rows])
    Xte = np.array([[r[f] for f in feats] for r in test_rows])
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xtr), ytr)
    return clf.predict_proba(scaler.transform(Xte))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    data = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    qrels_tr = load_qrels(data, "train")
    qrels_dev = load_qrels(data, "dev")

    train_rows = build(args.domain, "train", args.topk)
    dev_rows = build(args.domain, "dev", args.topk)
    for r in train_rows:
        r["label"] = 1.0 if r["doc"] in qrels_tr.get(r["qid"], {}) else 0.0
    for r in dev_rows:
        r["label"] = 1.0 if r["doc"] in qrels_dev.get(r["qid"], {}) else 0.0

    print(f"train rows {len(train_rows)} positives {sum(r['label'] for r in train_rows):.0f}")
    print(f"dev   rows {len(dev_rows)} positives {sum(r['label'] for r in dev_rows):.0f}")

    settings = {
        "dense_only": DENSE_FEATS,
        "dense_time": DENSE_FEATS + TIME_FEATS,
        "dense_ctrl": DENSE_FEATS + CTRL_FEATS,
        "dense_time_ctrl": DENSE_FEATS + TIME_FEATS + CTRL_FEATS,
    }
    out = {"domain": args.domain, "topk": args.topk, "runs": {}}
    for name, feats in settings.items():
        prob = fit_predict(train_rows, dev_rows, feats)
        by_q = {}
        for r, p in zip(dev_rows, prob):
            by_q.setdefault(r["qid"], []).append((r["doc"], float(p)))
        ranked = {q: sorted(v, key=lambda x: -x[1]) for q, v in by_q.items()}
        run_path = os.path.join(ROOT, "runs",
                                f"timefeat_{name}_{args.domain}_dev.trec")
        write_trec_run(run_path, ranked, tag=name)
        out["runs"][name] = run_path
        print(f"{name}: wrote {run_path}")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
