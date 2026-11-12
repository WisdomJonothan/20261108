#!/usr/bin/env python3
"""Generic zero-shot bi-encoder retrieval (for baseline comparisons).

Supports the pooling / prefix conventions of the models we compare:
  * bge-*        : CLS pooling, query instruction (model card)
  * gte-*        : CLS pooling, no instruction
  * mxbai-*      : mean pooling, query instruction (model card)
  * e5-*         : masked mean pooling, "query: " / "passage: " prefixes

No RETECO labels are read here. Embeddings are cached per (tag, corpus size).

Usage:
  python run_dense_generic.py --index-data <domain_dir> --split dev \
      --model Alibaba-NLP/gte-large-en-v1.5 --tag gtelarge --pooling cls \
      --work emb --out runs/dense_gtelarge_<domain>_dev.trec --meta ...
"""
import argparse
import json
import os
import platform
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_examples, read_jsonl, write_trec_run  # noqa: E402


def parse_dtype(name):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def _text_sha1(texts):
    import hashlib
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"||SEP||")
    return h.hexdigest()


def _dir_digest(path):
    """Content proxy for a model directory: relative path, size and mtime of
    every file under it. Cheap (no 1 GB read) and changes whenever the weights,
    tokenizer or config are replaced in place."""
    import hashlib
    h = hashlib.sha1()
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            f = os.path.join(root, name)
            try:
                st = os.stat(f)
            except OSError:
                continue
            h.update(os.path.relpath(f, path).encode("utf-8", "ignore"))
            h.update(b"|")
            h.update(str(st.st_size).encode())
            h.update(b"|")
            h.update(str(st.st_mtime_ns).encode())
            h.update(b"|")
    return h.hexdigest()[:16]


def _cache_fingerprint(model, pooling, max_length, dtype, prefix, texts):
    """Everything that changes the bytes of the .npy cache.

    The model enters as a content digest, not a path: a checkpoint retrained in
    place at the same path is a different encoder and must not reuse the cache.
    """
    import hashlib
    h = hashlib.sha1()
    for v in (model, _dir_digest(model), pooling, str(max_length), dtype, prefix):
        h.update(str(v).encode("utf-8", "ignore"))
        h.update(b"||SEP||")
    h.update(_text_sha1(texts).encode())
    return h.hexdigest()[:16]


def _check_cache(path, fp, adopt_legacy=False):
    """Fail closed on a proven mismatch, and on a cache with no provenance.

    A file without a sidecar cannot be shown to match the current inputs, so the
    default is to refuse it: reusing it silently is exactly how a stale
    embedding of a replaced model leaks into a run. `--adopt-legacy` adopts it
    once, writing a sidecar that records the adoption, and is opt-in.
    """
    meta = path + ".fp.json"
    if not os.path.isfile(meta):
        if not adopt_legacy:
            raise SystemExit(
                "[cache] %s has no fingerprint sidecar, so it cannot be shown to "
                "match the current model/texts; refusing to reuse. Delete the .npy "
                "or pass --refresh to rebuild, or --adopt-legacy to trust it once."
                % path)
        print(f"[cache] {os.path.basename(path)} has no fingerprint sidecar "
              f"(legacy); adopting it at the user's request")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fp, "provenance": "adopted-legacy"}, f, indent=1)
        return
    try:
        got = json.load(open(meta, encoding="utf-8")).get("fingerprint")
    except Exception:
        got = None
    if got != fp:
        raise SystemExit(
            "[cache] %s was built with fingerprint %s, current inputs give %s; "
            "refusing to reuse. Delete the .npy (and its .fp.json) or pass --refresh."
            % (path, got, fp))
    print(f"[cache] {os.path.basename(path)} fingerprint matches ({fp})")


def encode(texts, model, tok, device, batch_size, max_length, pooling, desc):
    embs, n = [], len(texts)
    t0 = time.perf_counter()
    # Character pre-truncation: the fast tokenizer tokenizes a whole string before
    # applying max_length, so one pathological document (hsm has a 44.5 M-character
    # file) makes it allocate gigabytes and die with a Rust OOM. A vocabulary token
    # is at most a few dozen characters, so 512 tokens always fit in 100x
    # max_length characters; the cap cannot change any embedding.
    char_cap = 100 * max_length
    for i in range(0, n, batch_size):
        batch = [t[:char_cap] for t in texts[i:i + batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        h = out.last_hidden_state
        if pooling == "cls":
            emb = h[:, 0]
        elif pooling == "mean":
            emb = (h * enc["attention_mask"].unsqueeze(-1)).sum(1) / \
                  enc["attention_mask"].sum(1, keepdim=True)
        else:  # last
            idx = enc["attention_mask"].sum(1) - 1
            emb = h[torch.arange(h.shape[0], device=h.device), idx]
        emb = torch.nn.functional.normalize(emb.float(), dim=-1)
        embs.append(emb.cpu().numpy())
        if i % (batch_size * 20) == 0:
            done = min(i + batch_size, n)
            print(f"  [{desc}] {done}/{n} texts  {done/(time.perf_counter()-t0):.1f} doc/s",
                  flush=True)
    return np.concatenate(embs, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pooling", choices=["cls", "mean", "last"], default="cls")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore an existing doc-embedding cache")
    ap.add_argument("--adopt-legacy", action="store_true",
                    help="trust a .npy cache that has no fingerprint sidecar "
                         "(default: refuse it)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--examples", default=None,
                    help="override examples_<split>.jsonl (test phase / arbitrary queries)")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dense] device={device} model={args.model} pooling={args.pooling}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = model.to(device).eval()
    if device == "cuda" and args.dtype != "fp32":
        model = model.to(parse_dtype(args.dtype))

    doc_ids, texts = load_corpus(args.index_data)
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.tag)
    doc_cache = os.path.join(args.work, f"doc_emb_{tag}_{len(doc_ids)}.npy")
    emb_fp = _cache_fingerprint(args.model, args.pooling, args.max_length, args.dtype,
                                args.doc_prefix, texts)
    if os.path.isfile(doc_cache) and not args.refresh:
        _check_cache(doc_cache, emb_fp, args.adopt_legacy)
        print(f"[dense] reusing {doc_cache}")
        doc_emb = np.load(doc_cache)
    else:
        t0 = time.perf_counter()
        doc_emb = encode([args.doc_prefix + t for t in texts], model, tok, device,
                         args.batch_size, args.max_length, args.pooling, "docs")
        print(f"[dense] encoded {len(doc_ids)} docs in {time.perf_counter()-t0:.1f}s")
        np.save(doc_cache, doc_emb)
        with open(doc_cache + ".fp.json", "w", encoding="utf-8") as f:
            json.dump({"fingerprint": emb_fp, "model": args.model, "pooling": args.pooling,
                       "max_length": args.max_length, "dtype": args.dtype,
                       "doc_prefix": args.doc_prefix, "n_docs": len(doc_ids)}, f, indent=1)

    if args.examples:
        recs = read_jsonl(args.examples)
        qids, queries = [r["id"] for r in recs], [r["query"] for r in recs]
    else:
        qids, queries = load_examples(args.index_data, args.split)
    q_texts = [args.query_prefix + q for q in queries]
    q_emb = encode(q_texts, model, tok, device, 16, args.max_length, args.pooling, "queries")

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
    write_trec_run(args.out, ranked, tag=tag)

    meta = {
        "system": f"zero-shot dense bi-encoder {args.model} ({args.pooling} pooling)",
        "model": args.model, "tag": args.tag, "pooling": args.pooling,
        "query_prefix": args.query_prefix, "doc_prefix": args.doc_prefix,
        "max_length": args.max_length, "dtype": args.dtype,
        "split": args.split, "num_docs": len(doc_ids), "num_queries": len(qids),
        "data": os.path.abspath(args.index_data), "device": device,
        "search_latency_ms": {"mean": sum(latencies) / len(latencies)},
        "embedding_dim": int(doc_emb.shape[1]),
        "doc_emb_file": os.path.abspath(doc_cache),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "python": platform.python_version(),
    }
    if device == "cuda":
        meta["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[dense] {len(qids)} queries, mean latency {meta['search_latency_ms']['mean']:.1f} ms")


if __name__ == "__main__":
    main()
