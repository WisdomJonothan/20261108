"""Listwise re-ranking with a local instruction-tuned LLM (RankGPT-style).

For each query: take the top-k candidates of a frozen run, present them to the
LLM as a numbered list, ask for a permutation, parse the ids, and write a new
run restricted to those candidates. Parse failures fall back to the original
order. Candidate input order is shuffled with a fixed per-query seed so the
contribution of position bias is at least controlable; the original order is
restored in the output numbering.

No qrels are read.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, read_trec_run, write_trec_run  # noqa: E402

ROOT = r"D:\RETECO"

PROMPT = """You are an expert search relevance annotator. Given a query and {n} candidate documents, rank them from most to least relevant to the query. Judge relevance by how well a document helps answer the query, not by word overlap.

Query: {query}

Candidates:
{docs}

Output only the ranking, using the candidate numbers in square brackets, from most to least relevant, separated by " > ". Use every number exactly once. Example: [3] > [1] > [2]"""

DOC_TMPL = "[{i}] {text}"


def clean(text, limit=700):
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = t.replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def load_queries(domain, split, mode="title"):
    q = {}
    path = os.path.join(ROOT, "data", "track1_tempo", domain, f"examples_{split}.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q[r["id"]] = r["query"]
    if mode == "title":
        return {k: clean(v.split("<p>")[0]) or clean(v) for k, v in q.items()}
    return {k: clean(v) for k, v in q.items()}


def parse_ranking(text, n):
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    seen, order = set(), []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            order.append(x)
    if len(order) != n:
        return None
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--model", default=r"D:\RETECO\models\Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--snippet-chars", type=int, default=700)
    ap.add_argument("--limit-queries", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device).eval()

    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    queries = load_queries(args.domain, args.split)
    cand = read_trec_run(args.run)
    qids = list(cand)[: args.limit_queries] if args.limit_queries else list(cand)

    out, per_q, failures, n_tok = {}, [], 0, 0
    for i, qid in enumerate(qids):
        docs = [t[2] for t in cand[qid][: args.k]]
        n = len(docs)
        # deterministic shuffle of the presented order, so position bias is not
        # aligned with the input ranking
        seed = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16)
        perm = np.random.RandomState(seed).permutation(n)
        presented = [docs[j] for j in perm]
        lines = [DOC_TMPL.format(i=j + 1, text=clean(corpus.get(d, ""), args.snippet_chars))
                 for j, d in enumerate(presented)]
        prompt = PROMPT.format(n=n, query=queries[qid], docs="\n".join(lines))
        msgs = [{"role": "user", "content": prompt}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt", truncation=True, max_length=4096).to(device)
        tq = time.time()
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        n_tok += int(gen.shape[1] - enc["input_ids"].shape[1])
        order = parse_ranking(txt, n)
        if order is None:
            failures += 1
            ranked = docs  # fallback: keep the first-stage order
        else:
            ranked = [presented[j - 1] for j in order]
        out[qid] = [(d, float(n - r)) for r, d in enumerate(ranked)]
        per_q.append(time.time() - tq)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(qids)} queries, failures={failures}", flush=True)

    write_trec_run(args.out, out, tag="listwise")
    lat = np.asarray(per_q)
    meta = {
        "kind": "listwise-llm", "domain": args.domain, "split": args.split,
        "candidate_run": os.path.basename(args.run), "candidate_k": args.k,
        "model": os.path.basename(args.model), "device": device,
        "max_new_tokens": args.max_new_tokens, "snippet_chars": args.snippet_chars,
        "shuffle": "per-query seeded permutation of the presented order",
        "queries": len(qids), "parse_failures": failures,
        "parse_failure_rate": round(failures / max(1, len(qids)), 4),
        "generated_tokens": n_tok,
        "seconds": round(time.time() - t0, 1),
        "latency_s_per_query": {
            "mean": round(float(lat.mean()), 2),
            "p50": round(float(np.percentile(lat, 50)), 2),
            "max": round(float(lat.max()), 2)},
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["latency_s_per_query"]), "failures", failures)


if __name__ == "__main__":
    main()
