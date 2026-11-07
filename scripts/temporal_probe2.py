"""Temporal-consistency probe v2, built to answer the protocol critique.

Adds to v1:
  * placebo control — the temporal statement of a *different* query is used
    (deterministic derangement). If the placebo scores as well as the real
    statement, the signal is topical, not temporal;
  * per-query AUC distribution and a query-clustered bootstrap CI for the
    pooled AUC and for the paired ΔAUC (dense vs dense+temporal);
  * collinearity check (correlation of the temporal score with dense rank and
    lexical overlap);
  * no statement ever sees qrels: statements come from guidance only.

Usage:
  python temporal_probe2.py --domain bitcoin --split dev --json out.json [--placebo]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import (load_corpus, load_examples, load_qrels,  # noqa: E402
                        read_trec_run)

ROOT = r"D:\RETECO"
YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
TOK = re.compile(r"[A-Za-z0-9]+")


def load_guidance(domain, split):
    p = os.path.join(ROOT, "data", "track1_tempo", domain, f"guidance_{split}.jsonl")
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["query_guidance"]
    return out


def statement(qg):
    ev = " ".join(qg.get("query_temporal_events", []))
    anchors = ", ".join(qg.get("key_time_anchors", []))
    summary = qg.get("query_summary", "")
    parts = []
    if ev:
        parts.append(f"This passage is about {ev}.")
    if anchors:
        parts.append(f"The relevant time period or condition is {anchors}.")
    if summary:
        parts.append(summary)
    return " ".join(parts)


def cluster_bootstrap(values_by_query, n_boot=2000, seed=0):
    """Bootstrap over queries; returns 2.5/97.5 percentiles of the pooled mean."""
    rng = np.random.default_rng(seed)
    qs = list(values_by_query)
    arr = [values_by_query[q] for q in qs]
    out = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        out.append(float(np.mean([arr[i] for i in idx])))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--placebo", action="store_true")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    data = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    ids, texts = load_corpus(data)
    corpus = dict(zip(ids, texts))
    id_index = {d: i for i, d in enumerate(ids)}
    qrels = load_qrels(data, args.split)
    qids, queries = load_examples(data, args.split)
    qmap = dict(zip(qids, queries))
    guidance = load_guidance(args.domain, args.split)
    run = read_trec_run(os.path.join(ROOT, "runs",
                                     f"dense_bge_{args.domain}_{args.split}.trec"))
    emb = np.load(os.path.join(ROOT, "emb",
                               f"doc_emb_bge-base-en-v1.5_{len(ids)}.npy"))

    from transformers import AutoModel, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models", "bge-base-en-v1.5"))
    model = AutoModel.from_pretrained(os.path.join(ROOT, "models", "bge-base-en-v1.5"),
                                      torch_dtype=torch.float16).eval().cuda()
    instr = "Represent this sentence for searching relevant passages: "
    qid_list = [q for q in qids if q in run and q in guidance]

    # statements, optionally replaced by another query's statement (placebo)
    stmt_qid = list(qid_list)
    if args.placebo:
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(stmt_qid))
        while any(perm[i] == i for i in range(len(stmt_qid))):
            perm = rng.permutation(len(stmt_qid))
        stmt_qid = [qid_list[i] for i in perm]
    stmts = [instr + statement(guidance[q]) for q in stmt_qid]

    stmt_emb = np.zeros((len(stmts), emb.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(stmts), 8):
            enc = tok(stmts[i:i + 8], padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state[:, 0].float()
            stmt_emb[i:i + 8] = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
    del model
    torch.cuda.empty_cache()

    rows = []
    for qi, qid in enumerate(qid_list):
        docs = [t[2] for t in run[qid][: args.topk]]
        gold = set(qrels.get(qid, {}))
        qtok = set(TOK.findall(qmap[qid].lower()))
        for rank, d in enumerate(docs, 1):
            de = emb[id_index[d]]
            cos = float(np.dot(stmt_emb[qi], de) /
                        (np.linalg.norm(stmt_emb[qi]) * np.linalg.norm(de) + 1e-9))
            dtok = TOK.findall(corpus.get(d, "").lower())
            rows.append({"qid": qid, "doc": d, "label": 1.0 if d in gold else 0.0,
                         "dense_rank": rank, "temporal_cos": cos,
                         "lex_overlap": len(qtok & set(dtok)) / max(1, len(qtok)),
                         "doc_len": float(np.log1p(len(dtok)))})

    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y = np.array([r["label"] for r in rows])
    groups = np.array([r["qid"] for r in rows])
    single = {
        "dense(-rank)": -np.array([r["dense_rank"] for r in rows]),
        "temporal_cos": np.array([r["temporal_cos"] for r in rows]),
        "lex_overlap": np.array([r["lex_overlap"] for r in rows]),
        "doc_len": np.array([r["doc_len"] for r in rows]),
    }
    aucs = {k: float(roc_auc_score(y, v)) for k, v in single.items()}

    def cv_oof(feats):
        X = np.array([[r[f] for f in feats] for r in rows])
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        return oof

    for r in rows:
        r["dense_z"] = -float(r["dense_rank"])
    oof_dense = cv_oof(["dense_z"])
    oof_plus = cv_oof(["dense_z", "temporal_cos"])
    auc_d = float(roc_auc_score(y, oof_dense))
    auc_p = float(roc_auc_score(y, oof_plus))

    # per-query AUCs
    per_q_dense, per_q_temp, per_q_plus, deltas = {}, {}, {}, {}
    for qid in qid_list:
        idx = [i for i, r in enumerate(rows) if r["qid"] == qid]
        ys = y[idx]
        if not (0 < ys.sum() < len(ys)):
            continue
        per_q_dense[qid] = float(roc_auc_score(ys, oof_dense[idx]))
        per_q_temp[qid] = float(roc_auc_score(ys, [rows[i]["temporal_cos"] for i in idx]))
        per_q_plus[qid] = float(roc_auc_score(ys, oof_plus[idx]))
        deltas[qid] = per_q_plus[qid] - per_q_dense[qid]

    corr = {
        "temporal_vs_dense_rank": float(np.corrcoef(
            [r["temporal_cos"] for r in rows], [r["dense_rank"] for r in rows])[0, 1]),
        "temporal_vs_lex_overlap": float(np.corrcoef(
            [r["temporal_cos"] for r in rows], [r["lex_overlap"] for r in rows])[0, 1]),
    }
    out = {
        "domain": args.domain, "split": args.split, "placebo": args.placebo,
        "rows": len(rows), "positives": int(y.sum()), "queries": len(qid_list),
        "auc_single": aucs,
        "auc_logistic_cv": {"dense_only": auc_d, "dense+temporal": auc_p,
                            "delta": auc_p - auc_d},
        "per_query_auc": {
            "n": len(per_q_temp),
            "dense_mean": float(np.mean(list(per_q_dense.values()))),
            "temporal_mean": float(np.mean(list(per_q_temp.values()))),
            "temporal_median": float(np.median(list(per_q_temp.values()))),
            "combined_mean": float(np.mean(list(per_q_plus.values()))),
            "delta_mean": float(np.mean(list(deltas.values()))),
            "delta_win_rate": float(np.mean([d > 0 for d in deltas.values()])),
            "delta_ci": cluster_bootstrap(deltas),
            "temporal_ci": cluster_bootstrap(per_q_temp),
        },
        "correlations": corr,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:2200])


if __name__ == "__main__":
    main()
