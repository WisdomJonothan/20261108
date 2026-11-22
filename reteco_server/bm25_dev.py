"""Per-domain BM25 over the RETECO corpus, fused with the frozen system.

Why: 138 of 519 dev queries return nothing relevant in the top ten and 52 of those have
no relevant document anywhere in the delivered top-100, so the missing signal is in the
first stage. The dense encoder is semantic; a lexical scorer is a cheap orthogonal view
and the organisers' BM25 baseline shows the corpus is answerable lexically.

The script writes, per domain, a BM25 run over the whole domain corpus and the fused runs
for a fixed weight grid, then reports domain-macro nDCG@10 for every arm. Weights are NOT
chosen on dev: the grid is reported in full and the selection (if any) happens on the
train folds.

Usage (server):
  .venv/bin/python scripts/bm25_dev.py --root . --k1 0.9 --b 0.4 --out results/analysis/bm25_dev.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

DOMAINS = [
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
    "law", "monero", "politics", "quant", "travel", "workplace",
]
TOKEN = re.compile(r"[a-z0-9][a-z0-9_\-]+")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docid = parts[0], parts[1], parts[2]
            score = float(parts[4]) if len(parts) >= 6 else float(parts[3])
            out.setdefault(qid, []).append((docid, score))
    for v in out.values():
        v.sort(key=lambda x: -x[1])
    return out


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            docid = parts[1] if len(parts) == 3 else parts[2]
            grade = int(float(parts[-1]))
            if grade > 0:
                out.setdefault(qid, {})[docid] = grade
    return out


def ndcg10(ranked: list[str], gold: dict[str, int]) -> float:
    dcg = sum(gold.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(gold.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def zscore(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mu, sd = float(scores.mean()), float(scores.std())
    return (scores - mu) / sd if sd > 1e-9 else np.zeros_like(scores)


def bm25_scores(matrix, doc_len: np.ndarray, avg_len: float, q_idx: np.ndarray,
                idf: np.ndarray, k1: float, b: float) -> np.ndarray:
    """Score every document for one query's term indices."""
    rows = matrix[:, q_idx]                      # docs x query terms (binary counts)
    counts = np.asarray(rows.todense()) if hasattr(rows, "todense") else np.asarray(rows)
    counts = counts.astype(np.float64)
    denom = counts + k1 * (1.0 - b + b * doc_len / max(avg_len, 1e-9))[:, None]
    term = counts * (k1 + 1.0) / np.maximum(denom, 1e-9) * idf[q_idx][None, :]
    return term.sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--weights", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--out", default="results/analysis/bm25_dev.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    weights = [float(w) for w in args.weights.split(",")]

    per_domain: dict[str, dict] = {}
    arm_scores: dict[str, dict[str, list[float]]] = {f"bm25@{w}": {} for w in weights}
    arm_scores["bm25_only"] = {}
    arm_scores["frozen"] = {}

    for dom in DOMAINS:
        base = root / "data" / "track1_tempo" / dom
        docs = [{"id": d["id"], "text": d.get("content") or d.get("text") or ""}
                for d in read_jsonl(base / "documents.jsonl")]
        gold = read_qrels(base / f"qrels_{args.split}.txt")
        frozen = read_run(root / "runs" / f"frozen_fo_{dom}_{args.split}.trec") \
            if (root / "runs" / f"frozen_fo_{dom}_{args.split}.trec").exists() else \
            read_run(root / "runs" / f"sub_sft_e2bc_{dom}_{args.split}.trec")
        vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=2,
                              dtype=np.int32)
        matrix = vec.fit_transform([d["text"] for d in docs]).tocsr()
        vocab = vec.vocabulary_
        doc_len = np.asarray(matrix.sum(axis=1)).ravel()
        avg_len = float(doc_len.mean())
        doc_ids = [d["id"] for d in docs]
        n_docs = len(doc_ids)
        df = np.asarray((matrix > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        query_text = {q["id"]: (q.get("query") or q.get("text") or "")
                      for q in read_jsonl(base / f"examples_{args.split}.jsonl")}
        bm25_run: dict[str, list[tuple[str, float]]] = {}
        for qid, text in query_text.items():
            terms = [t for t in re.findall(r"[a-z0-9][a-z0-9_\-]+", text.lower()) if t in vocab]
            if not terms:
                bm25_run[qid] = []
                continue
            q_idx = np.array([vocab[t] for t in terms], dtype=np.int64)
            scores = bm25_scores(matrix, doc_len, avg_len, q_idx, idf, args.k1, args.b)
            order = np.argsort(-scores)[:100]
            bm25_run[qid] = [(doc_ids[i], float(scores[i])) for i in order]

        scores = {a: [] for a in arm_scores}
        for qid, g in gold.items():
            fr = frozen.get(qid, [])
            br = bm25_run.get(qid, [])
            if not fr:
                continue
            fmap = dict(fr)
            fz = {d: s for d, s in zip([d for d, _ in fr], zscore(np.array([s for _, s in fr])))}
            fz = dict(zip([d for d, _ in fr], zscore(np.array([s for _, s in fr]))))
            bz = dict(zip([d for d, _ in br], zscore(np.array([s for _, s in br]))) ) if br else {}
            base_rank = [d for d, _ in fr][:10]
            scores["frozen"].append(ndcg10(base_rank, g))
            b_only = [d for d, _ in br][:10]
            scores["bm25_only"].append(ndcg10(b_only, g))
            for w in weights:
                combined = {}
                for d in set(list(fz) + list(bz)):
                    combined[d] = fz.get(d, -3.0) + w * bz.get(d, -3.0)
                ranked = [d for d, _ in sorted(combined.items(), key=lambda kv: -kv[1])][:10]
                scores[f"bm25@{w}"].append(ndcg10(ranked, g))
        per_domain[dom] = {a: float(np.mean(v)) if v else float("nan") for a, v in scores.items()}
        for a in arm_scores:
            if scores[a]:
                arm_scores[a][dom] = per_domain[dom][a]
        print(dom, {a: round(per_domain[dom][a], 4) for a in per_domain[dom]}, flush=True)

    summary = {a: float(np.mean(list(v.values()))) for a, v in arm_scores.items() if v}
    payload = {"split": args.split, "k1": args.k1, "b": args.b, "weights": weights,
               "domain_macro": summary, "per_domain": per_domain}
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("SUMMARY", {k: round(v, 4) for k, v in summary.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
