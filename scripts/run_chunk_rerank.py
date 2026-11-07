"""Chunk-aware cross-encoder re-ranking over a fixed candidate run.

Why this script exists: `run_rerank.py` passes (query, document) to a 512-token
encoder with the default `longest_first` truncation. RETECO queries are long
(median 221 tokens, max 1017), so the document side was silently cut to
~250 tokens on average and to almost nothing for the longest queries. That
confounds any conclusion about the re-ranker.

Here the query is capped explicitly (`--q-cap`) and the document is split into
fixed windows (`--win`, `--stride`); every (query, window) pair is scored and
the per-document score is pooled (`max` or `mean`). Candidate containment is
unchanged: output documents are exactly a subset of the input top-k.

No qrels are read.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, read_trec_run, write_trec_run  # noqa: E402

ROOT = r"D:\RETECO"


def windows_of(tok, text, win, stride):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= win:
        return [tok.decode(ids)] if ids else [""]
    out = []
    for start in range(0, len(ids), stride):
        chunk = ids[start:start + win]
        if not chunk:
            break
        out.append(tok.decode(chunk))
        if start + win >= len(ids):
            break
    return out


def load_queries(domain, split, mode="full"):
    """mode: full | title | summary | clip (clip = strip then cap by tokens)."""
    import re

    q = {}
    with open(os.path.join(ROOT, "data", "track1_tempo", domain,
                           f"examples_{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q[r["id"]] = r["query"]
    def clean(t):
        t = re.sub(r"<[^>]+>", " ", t or "")
        t = t.replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
        return re.sub(r"\s+", " ", t).strip()
    if mode == "title":
        return {k: clean(v.split("<p>")[0]) or clean(v) for k, v in q.items()}
    if mode == "summary":
        path = os.path.join(ROOT, "data", "track1_tempo", domain,
                            f"guidance_{split}.jsonl")
        out = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = clean(r["query_guidance"].get("query_summary", ""))
        return {k: out.get(k) or clean(v.split("<p>")[0]) for k, v in q.items()}
    if mode == "clip":
        return {k: clean(v)[:800] for k, v in q.items()}
    return {k: clean(v) for k, v in q.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--q-cap", type=int, default=128)
    ap.add_argument("--win", type=int, default=384)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--pool", choices=["max", "mean"], default="max")
    ap.add_argument("--model", default=r"D:\RETECO\models\bge-reranker-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--q-mode", choices=["full", "title", "summary", "clip"], default="full")
    args = ap.parse_args()

    assert args.q_cap + args.win <= 508, "query cap + window must fit in 512 tokens"
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if (args.fp32 or device != "cuda") else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=dtype).eval().to(device)

    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    queries = load_queries(args.domain, args.split, args.q_mode)
    cand = read_trec_run(args.run)

    results, per_q_ms, n_pairs, n_windows, n_missing = {}, [], 0, 0, 0
    with torch.inference_mode():
        for qi, qid in enumerate(cand):
            q_ids = tok(queries[qid], add_special_tokens=False,
                        truncation=True, max_length=args.q_cap)["input_ids"]
            q_text = tok.decode(q_ids)
            docs = [t[2] for t in cand[qid][: args.k]]
            flat, spans = [], []
            for d in docs:
                c = corpus.get(d)
                if c is None:
                    n_missing += 1
                    c = ""
                ws = windows_of(tok, c, args.win, args.stride)
                spans.append((len(flat), len(flat) + len(ws)))
                flat.extend(ws)
                n_windows += len(ws)
            scores = []
            tq = time.time()
            for i in range(0, len(flat), args.batch):
                batch = flat[i : i + args.batch]
                enc = tok([q_text] * len(batch), batch, padding=True,
                          truncation=True, max_length=args.q_cap + args.win + 4,
                          return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.view(-1).float().cpu().numpy()
                scores.extend(logits.tolist())
                n_pairs += len(batch)
            scores = np.asarray(scores)
            doc_scores = []
            for s, e in spans:
                seg = scores[s:e]
                doc_scores.append(float(seg.max() if args.pool == "max" else seg.mean()))
            per_q_ms.append((time.time() - tq) * 1000.0)
            order = np.argsort(-np.asarray(doc_scores))[: args.k]
            results[qid] = [(docs[j], float(doc_scores[j])) for j in order]
            if (qi + 1) % 10 == 0:
                print(f"  {qi+1}/{len(cand)} queries", flush=True)

    write_trec_run(args.out, results, tag="chunkrerank")
    lat = np.asarray(per_q_ms)
    meta = {
        "domain": args.domain, "split": args.split,
        "candidate_run": os.path.basename(args.run), "candidate_k": args.k,
        "model": os.path.basename(args.model), "dtype": str(dtype).replace("torch.", ""),
        "q_mode": args.q_mode, "q_cap": args.q_cap, "window": args.win, "stride": args.stride,
        "pool": args.pool, "batch": args.batch,
        "queries": len(cand), "pairs_scored": n_pairs, "windows": n_windows,
        "mean_windows_per_doc": round(n_windows / max(1, args.k * len(cand)), 2),
        "missing_docs": n_missing,
        "seconds": round(time.time() - t0, 1),
        "pairs_per_s": round(n_pairs / max(1e-9, time.time() - t0), 1),
        "latency_ms_per_query": {
            "mean": round(float(lat.mean()), 1),
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
        },
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device == "cuda" else None,
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["latency_ms_per_query"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
