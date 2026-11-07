"""Diagnostics: document length vs cross-encoder behaviour on the candidate set.

Answers three questions for one domain/split:
  1. how often does a candidate exceed the 512-token limit of the reranker?
  2. are gold documents longer or shorter than the rest?
  3. does the cross-encoder systematically demote long documents?
"""
import argparse
import json
import os
ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_qrels, read_trec_run  # noqa: E402


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bitcoin")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dense", required=True)
    ap.add_argument("--rerank", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--tokenizer", default=r"D:\RETECO\models\bge-reranker-base")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    qrels = load_qrels(os.path.join(ROOT, "data", "track1_tempo", args.domain), args.split)
    dense = read_trec_run(args.dense)
    rer = read_trec_run(args.rerank)

    lengths, trunc, gold_len, nongold_len = [], 0, [], []
    rank_pairs = []  # (length, dense_rank - rerank_rank)  positive => promoted
    per_query = {}

    for qid, items in dense.items():
        gold = set(qrels.get(qid, {}))
        docs = [item[2] for item in items[: args.k]]
        d_rank = {d: i + 1 for i, d in enumerate(docs)}
        r_rank = {item[2]: item[0] for item in rer.get(qid, [])}
        for d in docs:
            t = corpus.get(d, "")
            n = len(tok(t, add_special_tokens=False)["input_ids"])
            lengths.append(n)
            if n > 510:
                trunc += 1
            (gold_len if d in gold else nongold_len).append(n)
            if d in r_rank:
                rank_pairs.append((n, d_rank[d] - r_rank[d]))
        per_query[qid] = {
            "n_cand": len(docs),
            "n_truncated": sum(
                1 for d in docs
                if len(tok(corpus.get(d, ""), add_special_tokens=False)["input_ids"]) > 510),
            "gold": len(gold),
            "gold_in_cand": len([d for d in docs if d in gold]),
        }

    arr = np.array(lengths)
    out = {
        "domain": args.domain,
        "split": args.split,
        "candidates": len(lengths),
        "tokens_mean": float(arr.mean()),
        "tokens_p50": float(np.percentile(arr, 50)),
        "tokens_p90": float(np.percentile(arr, 90)),
        "tokens_p99": float(np.percentile(arr, 99)),
        "tokens_max": int(arr.max()),
        "pct_over_510_tokens": 100.0 * trunc / len(lengths),
        "gold_tokens_mean": statistics.mean(gold_len) if gold_len else None,
        "nongold_tokens_mean": statistics.mean(nongold_len) if nongold_len else None,
        "gold_count": len(gold_len),
        "spearman_len_vs_rank_gain": spearman(
            [p[0] for p in rank_pairs], [p[1] for p in rank_pairs]),
        "rank_gain_long_gt510": statistics.mean(
            [p[1] for p in rank_pairs if p[0] > 510]) if any(p[0] > 510 for p in rank_pairs) else None,
        "rank_gain_short_le510": statistics.mean(
            [p[1] for p in rank_pairs if p[0] <= 510]) if any(p[0] <= 510 for p in rank_pairs) else None,
        "per_query": per_query,
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in out.items() if k != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
