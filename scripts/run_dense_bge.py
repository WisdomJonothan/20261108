#!/usr/bin/env python3
"""Phase-0 dense baseline: BAAI/bge-base-en-v1.5 zero-shot bi-encoder retrieval.

No RETECO training data is used: the encoder is used exactly as released, with
the model card's query instruction. Embeddings are written to disk so that every
run can be re-scored without re-encoding.

Usage:
  python run_dense_bge.py --index-data D:/RETECO/data/track1_tempo/iota \
      --split dev --work D:/RETECO/emb --out ... --meta ...
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from lib_reteco import load_corpus, load_examples, write_trec_run

Q_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def encode(texts, model, tok, device, batch_size, max_length, desc):
    embs = []
    n = len(texts)
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        # BGE uses CLS pooling
        emb = out.last_hidden_state[:, 0]
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        embs.append(emb.cpu().numpy())
        if i % (batch_size * 20) == 0:
            done = min(i + batch_size, n)
            rate = done / (time.perf_counter() - t0)
            print(f"  [{desc}] {done}/{n} docs  {rate:.1f} doc/s", flush=True)
    return np.concatenate(embs, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--topk", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dense] device={device} model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    if device == "cuda":
        model = model.half()

    doc_ids, texts = load_corpus(args.index_data)
    import re

    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(args.model.rstrip("/\\")))
    doc_cache = os.path.join(args.work, f"doc_emb_{tag}_{len(doc_ids)}.npy")
    if os.path.isfile(doc_cache):
        print(f"[dense] reusing cached doc embeddings {doc_cache}")
        doc_emb = np.load(doc_cache)
    else:
        t0 = time.perf_counter()
        doc_emb = encode(texts, model, tok, device, args.batch_size, args.max_length, "docs")
        print(f"[dense] encoded {len(doc_ids)} docs in {time.perf_counter()-t0:.1f}s")
        np.save(doc_cache, doc_emb)

    qids, queries = load_examples(args.index_data, args.split)
    q_texts = [Q_INSTRUCTION + q for q in queries]
    t0 = time.perf_counter()
    q_emb = encode(q_texts, model, tok, device, 16, args.max_length, "queries")
    q_time = time.perf_counter() - t0

    # exact search (cosine == dot on normalised vectors), chunked over queries
    doc_t = torch.from_numpy(doc_emb).to(device)
    ranked, latencies = {}, []
    for i, qid in enumerate(qids):
        q = torch.from_numpy(q_emb[i]).to(device)
        ts = time.perf_counter()
        sims = doc_t @ q
        k = min(args.topk, sims.shape[0])
        vals, idx = torch.topk(sims, k)
        latencies.append((time.perf_counter() - ts) * 1000.0)
        ranked[qid] = [(doc_ids[j], float(v)) for j, v in zip(idx.tolist(), vals.tolist())]

    write_trec_run(args.out, ranked, tag="densebge")

    lat = sorted(latencies)
    meta = {
        "system": f"zero-shot dense bi-encoder {args.model} (CLS pooling, cosine)",
        "model": args.model,
        "query_instruction": Q_INSTRUCTION,
        "query_instruction_source": "BGE v1.5 model card recommendation",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "num_docs": len(doc_ids),
        "num_queries": len(qids),
        "split": args.split,
        "data": os.path.abspath(args.index_data),
        "device": device,
        "dtype": "float16" if device == "cuda" else "float32",
        "seconds_query_encode": q_time,
        "search_latency_ms": {"mean": sum(latency := latencies) / len(latencies),
                              "median_of_means": lat[len(lat) // 2]},
        "embedding_dim": int(doc_emb.shape[1]),
        "doc_emb_file": os.path.abspath(doc_cache),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
        "run_file": os.path.abspath(args.out),
    }
    if device == "cuda":
        meta["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[dense] {len(qids)} queries, latency mean "
          f"{meta['search_latency_ms']['mean']:.1f} ms")


if __name__ == "__main__":
    main()
