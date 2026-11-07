"""Pairwise LLM re-ranking (round-robin) over the dense top-K.

Motivation: the small local instruct model cannot produce a listwise permutation
(it echoes the input order), but a direct probe shows it *can* compare two
passages (71.7% accuracy with balanced position bias). This script turns that
signal into a ranking by running all pairwise comparisons among the top-K
candidates, counting wins, and reordering by win count.

Cost is quadratic: K=10 -> 45 calls per query.

No qrels are read.
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, read_trec_run, write_trec_run  # noqa: E402

ROOT = r"D:\RETECO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "qwen"))
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).eval().cuda()

    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    queries = {}
    with open(os.path.join(ROOT, "data", "track1_tempo", args.domain,
                           f"examples_{args.split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["id"]] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ",
                                                          r["query"])).strip()

    def ask(prompt):
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt", truncation=True,
                  max_length=3000).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    cand = read_trec_run(args.run)
    qids = list(cand)
    if args.limit_queries:
        qids = qids[: args.limit_queries]

    results, lat, calls, parse_fail = {}, [], 0, 0
    for qi, qid in enumerate(qids):
        tq = time.time()
        docs = [t[2] for t in cand[qid][: args.k]]
        wins = {d: 0 for d in docs}
        n = len(docs)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = docs[i], docs[j]
                if (hash((qid, a, b)) & 1):  # deterministic order randomisation
                    a, b = b, a
                prompt = (f"Search query: {queries[qid]}\n\n"
                          f"[A] {corpus.get(a, '')[: args.chars]}\n\n"
                          f"[B] {corpus.get(b, '')[: args.chars]}\n\n"
                          "Which passage is more relevant to the search query? "
                          "Answer with A or B only.")
                ans = ask(prompt).strip()[:10]
                calls += 1
                if re.search(r"\bA\b", ans):
                    wins[a] += 1
                elif re.search(r"\bB\b", ans):
                    wins[b] += 1
                else:
                    parse_fail += 1
                    wins[a] += 0.5
                    wins[b] += 0.5
        order = sorted(docs, key=lambda d: (-wins[d], docs.index(d)))
        results[qid] = [(d, float(n - r)) for r, d in enumerate(order)]
        lat.append((time.time() - tq) * 1000.0)
        print(f"  {qi+1}/{len(qids)} queries", flush=True)

    write_trec_run(args.out, results, tag="pairwise")
    meta = {
        "domain": args.domain, "split": args.split,
        "candidate_run": os.path.basename(args.run), "candidate_k": args.k,
        "model": os.path.basename(args.model.rstrip("/\\")),
        "queries": len(qids), "llm_calls": calls,
        "parse_failures": parse_fail,
        "seconds": round(time.time() - t0, 1),
        "latency_ms_per_query": {
            "mean": round(sum(lat) / len(lat), 1),
            "p50": round(sorted(lat)[len(lat) // 2], 1),
            "max": round(max(lat), 1),
        },
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["latency_ms_per_query"]))


if __name__ == "__main__":
    main()
