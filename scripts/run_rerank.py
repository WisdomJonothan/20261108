"""Cross-encoder (BGE reranker) re-ranking over a fixed candidate run.

Reads the top-k candidates of an existing TREC run, re-scores every
(query, candidate-document) pair with a pretrained cross-encoder, and writes a
new TREC run restricted to those candidates.  The candidate budget is therefore
exactly the input budget: no document outside it can enter the new run.

The corpus is read from documents.jsonl (only id and content exist there); the
model never sees qrels or gold ids.
"""
import argparse
import importlib.metadata as md
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, read_trec_run, write_trec_run  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True, help="candidate run (trec)")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--model", default=r"D:\RETECO\models\bge-reranker-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--fp32", action="store_true", help="force float32 (default fp16 on GPU)")
    args = ap.parse_args()

    t0 = time.time()
    domain_dir = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    ids, texts = load_corpus(domain_dir)
    corpus = dict(zip(ids, texts))
    del ids, texts
    cand = read_trec_run(args.run)
    qids = list(cand)
    if args.limit_queries:
        qids = qids[: args.limit_queries]
    print(f"corpus {len(corpus)} docs | queries {len(qids)} | k={args.k}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if (args.fp32 or device != "cuda") else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    model.to(device)

    results, per_query_ms, n_pairs, n_missing = {}, [], 0, 0
    with torch.inference_mode():
        for qi, qid in enumerate(qids):
            docs = [t[2] for t in cand[qid][: args.k]]
            qtext = QUERIES[qid]
            texts = []
            for d in docs:
                c = corpus.get(d)
                if c is None:
                    n_missing += 1
                    c = ""
                texts.append(c)
            scores = []
            tq = time.time()
            for i in range(0, len(texts), args.batch):
                batch = texts[i : i + args.batch]
                enc = tok(
                    [qtext] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.view(-1).float().cpu().numpy()
                scores.extend(logits.tolist())
                n_pairs += len(batch)
            per_query_ms.append((time.time() - tq) * 1000.0)
            order = np.argsort(-np.asarray(scores))[: args.k]
            results[qid] = [(docs[j], float(scores[j])) for j in order]
            if (qi + 1) % 20 == 0:
                print(f"  {qi+1}/{len(qids)} queries", flush=True)

    write_trec_run(args.out, results, tag="rerank")
    lat = np.asarray(per_query_ms)
    meta = {
        "domain": args.domain,
        "split": args.split,
        "candidate_run": os.path.basename(args.run),
        "candidate_k": args.k,
        "model": args.model,
        "model_sha": hashlib_file(os.path.join(args.model, "model.safetensors")),
        "device": device,
        "dtype": str(dtype).replace("torch.",""),
        "batch": args.batch,
        "max_length": args.max_length,
        "queries": len(qids),
        "pairs_scored": n_pairs,
        "missing_docs": n_missing,
        "seconds": round(time.time() - t0, 2),
        "pairs_per_s": round(n_pairs / max(1e-9, (time.time() - t0)), 1),
        "latency_ms_per_query": {
            "mean": round(float(lat.mean()), 2),
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "max": round(float(lat.max()), 2),
        },
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device == "cuda"
        else None,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": md.version("transformers"),
        },
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["latency_ms_per_query"]))
    print(f"wrote {args.out}")


def hashlib_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


QUERIES = {}


def load_queries(domain: str, split: str) -> None:
    path = os.path.join(ROOT, "data", "track1_tempo", domain, f"examples_{split}.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            QUERIES[r["id"]] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r["query"])).strip()


if __name__ == "__main__":
    args_pre = sys.argv[1:]
    dom = args_pre[args_pre.index("--domain") + 1]
    spl = args_pre[args_pre.index("--split") + 1]
    load_queries(dom, spl)
    main()
