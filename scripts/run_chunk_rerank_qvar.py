"""Windowed cross-encoder re-ranking with the constraint-preserving query forms
added in ``train_ce.py`` (``title|body|full|ctxt|summ``).

``run_chunk_rerank.py`` only knows ``full|title|summary|clip``.  The query
representation study (ladder step 3) needs *the same* composer that the training
and validation code uses, otherwise the val numbers that pick a representation
would not transfer to dev.  This script therefore reuses
``train_ce.compose_query`` / ``split_raw_query`` and only re-implements the
scoring loop, which is a straight copy of ``run_chunk_rerank.main``.

Usage example (dev, 13 domains)::

    python scripts/run_chunk_rerank_qvar.py --domain iota --split dev \
        --run runs/dense_mxbai_iota_dev.trec --model results/ce_ft/lx_allpos/best \
        --q-mode ctxt --q-cap-words 64 --q-tokens 256 \
        --win 250 --stride 192 --pool max --batch 32 \
        --out runs/rerank_qv_ctxt_iota_dev.trec --meta runs/rerank_qv_ctxt_iota_dev.meta.json
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_chunk_rerank import (load_corpus, load_queries, read_trec_run,
                              windows_of, write_trec_run)


def _load_train_ce():
    """Import train_ce by path (it is a script, not a package module)."""
    path = os.path.join(ROOT, "scripts", "train_ce.py")
    spec = importlib.util.spec_from_file_location("train_ce_shared", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_raw_queries(domain, split, examples=None):
    """Raw query text, HTML intact.

    ``load_queries`` cleans the tags first, which destroys the ``<p>`` that
    separates title from body; splitting after cleaning silently turns the whole
    question into the "title" and empties the body. Splitting must happen on the
    raw string, exactly as ``train_ce`` does at training time.
    """
    import json as _json
    path = examples or os.path.join(ROOT, "data", "track1_tempo", domain,
                                    "examples_%s.jsonl" % split)
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = _json.loads(line)
            out[r["id"]] = r["query"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--examples", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--q-mode", required=True,
                    choices=["title", "body", "full", "ctxt", "summ"])
    ap.add_argument("--q-cap-words", type=int, default=64)
    ap.add_argument("--pair-max-len", type=int, default=0,
                    help="hard token ceiling for the (query, window) pair; 0 = "
                         "q_tokens + win + 4. Set it to the value the checkpoint "
                         "was validated with to reproduce training-time scores")
    ap.add_argument("--q-tokens", type=int, default=256,
                    help="hard token ceiling applied after composing the query")
    ap.add_argument("--win", type=int, default=250)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--pool", default="max", choices=["max", "mean"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    assert args.q_tokens + args.win <= 508, "query cap + window must fit in 512 tokens"
    t0 = time.time()
    tce = _load_train_ce()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if (args.fp32 or device != "cuda") else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, torch_dtype=dtype).eval().to(device)

    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    raw = load_raw_queries(args.domain, args.split, args.examples)
    queries = {}
    for qid, q in raw.items():
        t, b = tce.split_raw_query(q)
        queries[qid] = tce.compose_query(tce.clean_query(t), tce.clean_query(b),
                                         args.q_mode, args.q_cap_words)

    cand = read_trec_run(args.run)
    if args.examples:
        keep = [q for q in cand if q in queries]
        if len(keep) != len(cand):
            print("--examples: scoring %d/%d run topics" % (len(keep), len(cand)), flush=True)
        cand = {q: cand[q] for q in keep}
    if not cand:
        raise SystemExit("no queries to score after applying --examples")

    results, per_q_ms, n_pairs, n_windows, n_missing = {}, [], 0, 0, 0
    with torch.inference_mode():
        for qi, qid in enumerate(cand):
            q_ids = tok(queries[qid], add_special_tokens=False,
                        truncation=True, max_length=args.q_tokens)["input_ids"]
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
                batch = flat[i:i + args.batch]
                enc = tok([q_text] * len(batch), batch, padding=True,
                          truncation=True,
                          max_length=(args.pair_max_len or args.q_tokens + args.win + 4),
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
                print("  %d/%d queries" % (qi + 1, len(cand)), flush=True)

    write_trec_run(args.out, results, tag="chunkrerank")
    lat = np.asarray(per_q_ms)
    meta = {
        "domain": args.domain, "split": args.split,
        "candidate_run": os.path.basename(args.run), "candidate_k": args.k,
        "model": os.path.basename(args.model.rstrip("/")),
        "dtype": str(dtype).replace("torch.", ""),
        "q_mode": args.q_mode, "q_cap_words": args.q_cap_words,
        "q_tokens": args.q_tokens, "window": args.win, "stride": args.stride,
        "pool": args.pool, "batch": args.batch,
        "queries": len(cand), "pairs_scored": n_pairs, "windows": n_windows,
        "mean_windows_per_doc": round(n_windows / max(1, args.k * len(cand)), 2),
        "missing_docs": n_missing,
        "seconds": round(time.time() - t0, 1),
        "mean_ms_per_query": round(float(lat.mean()), 1) if len(lat) else None,
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
