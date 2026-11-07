"""Feasibility probe: is there a *semantic* temporal signal?

Question: given the organizers' temporal annotations (query_temporal_events,
key_time_anchors, query_summary), can we score whether a document actually
discusses the required period/event — and does that score separate gold from
non-gold documents *beyond* the dense topical score?

Construction:
  * temporal statement per query (from guidance);
  * score 1: cosine between the statement embedding and the document embedding
    (bge-base, the same encoder as the retriever);
  * score 2 (optional): a local instruct model answers yes/no to
    "does this passage concern <events/period>?"; score = P(Yes);
  * controls: dense score rank, document length, lexical overlap, year match.

Metrics: AUC of each score for gold vs non-gold among the dense top-100, and the
AUC of a logistic model with dense only vs dense + temporal features, with
query-grouped cross-validation (no query appears in both folds).

No test labels exist; qrels are used only as evaluation labels.
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


def statement(qg, query):
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
    return " ".join(parts) if parts else query


def doc_embeddings(domain, n_docs):
    cache = os.path.join(ROOT, "emb", f"doc_emb_bge-base-en-v1.5_{n_docs}.npy")
    return np.load(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--json", required=True)
    ap.add_argument("--llm", action="store_true", help="also run the local LLM yes/no probe")
    ap.add_argument("--llm-queries", type=int, default=10)
    ap.add_argument("--llm-docs", type=int, default=20)
    args = ap.parse_args()

    data = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    ids, texts = load_corpus(data)
    corpus = dict(zip(ids, texts))
    id_index = {d: i for i, d in enumerate(ids)}
    qrels = load_qrels(data, args.split)
    qids, queries = load_examples(data, args.split)
    guidance = load_guidance(args.domain, args.split)
    run = read_trec_run(os.path.join(ROOT, "runs",
                                     f"dense_bge_{args.domain}_{args.split}.trec"))
    emb = doc_embeddings(args.domain, len(ids))

    # encode temporal statements with the same bi-encoder
    from transformers import AutoModel, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models", "bge-base-en-v1.5"))
    model = AutoModel.from_pretrained(os.path.join(ROOT, "models", "bge-base-en-v1.5"),
                                      torch_dtype=torch.float16).eval().cuda()
    instr = "Represent this sentence for searching relevant passages: "
    qid_list = [q for q in qids if q in run and q in guidance]
    stmts = [instr + statement(guidance[q], dict(zip(qids, queries))[q]) for q in qid_list]
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
        qtok = set(TOK.findall(dict(zip(qids, queries))[qid].lower()))
        q_years = set(YEAR.findall(dict(zip(qids, queries))[qid]))
        for rank, d in enumerate(docs, 1):
            de = emb[id_index[d]]
            cos = float(np.dot(stmt_emb[qi], de) /
                        (np.linalg.norm(stmt_emb[qi]) * np.linalg.norm(de) + 1e-9))
            dtok = TOK.findall(corpus.get(d, "").lower())
            d_years = set(YEAR.findall(corpus.get(d, "")))
            rows.append({
                "qid": qid, "doc": d, "label": 1.0 if d in gold else 0.0,
                "dense_rank": rank, "temporal_cos": cos,
                "lex_overlap": len(qtok & set(dtok)) / max(1, len(qtok)),
                "year_match": 1.0 if (q_years & d_years) else 0.0,
                "doc_len": float(np.log1p(len(dtok))),
            })

    # AUCs
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y = np.array([r["label"] for r in rows])
    groups = np.array([r["qid"] for r in rows])
    scores = {
        "dense(-rank)": -np.array([r["dense_rank"] for r in rows]),
        "temporal_cos": np.array([r["temporal_cos"] for r in rows]),
        "lex_overlap": np.array([r["lex_overlap"] for r in rows]),
        "year_match": np.array([r["year_match"] for r in rows]),
        "doc_len": np.array([r["doc_len"] for r in rows]),
    }
    aucs = {k: float(roc_auc_score(y, v)) for k, v in scores.items()}

    def cv_auc(feats):
        X = np.array([[r[f] for f in feats] for r in rows])
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        return float(roc_auc_score(y, oof)), oof

    dense_cols = ["dense_rank"]
    for r in rows:
        r["dense_score_z"] = -float(r["dense_rank"])
    auc_dense, _ = cv_auc(["dense_score_z"])
    auc_plus, _ = cv_auc(["dense_score_z", "temporal_cos"])
    auc_plus_ctrl, _ = cv_auc(["dense_score_z", "temporal_cos", "doc_len", "lex_overlap"])

    out = {
        "domain": args.domain, "split": args.split, "rows": len(rows),
        "positives": int(y.sum()), "queries": len(qid_list),
        "auc_single": aucs,
        "auc_logistic_cv": {"dense_only": auc_dense,
                            "dense+temporal": auc_plus,
                            "dense+temporal+controls": auc_plus_ctrl},
        "per_query_auc": {},
    }
    # per-query AUC of temporal_cos (only queries with both classes)
    for qid in qid_list:
        sub = [r for r in rows if r["qid"] == qid]
        ys = [r["label"] for r in sub]
        if 0 < sum(ys) < len(ys):
            out["per_query_auc"][qid] = float(
                roc_auc_score(ys, [r["temporal_cos"] for r in sub]))
    vals = list(out["per_query_auc"].values())
    out["per_query_auc_mean"] = float(np.mean(vals)) if vals else None
    out["per_query_auc_n"] = len(vals)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_query_auc",)}, indent=2)[:1600])


if __name__ == "__main__":
    main()
