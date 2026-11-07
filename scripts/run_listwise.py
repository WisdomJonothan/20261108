"""Listwise LLM re-ranking (RankGPT-style) over a frozen candidate run.

Reads the top-k candidates of an existing TREC run, asks a local instruct model
to order them by relevance, and writes a new run restricted to the same
candidates (recall is unchanged by construction).

Protocol
--------
* the candidate list is shown as numbered passages (first `--passage-chars`
  characters of each document, matching RankGPT's passage-summary setup);
* the model must answer with a permutation of the numbers;
* sliding window (window `--window`, stride `--stride`), applied from the bottom
  of the list upwards, as in RankGPT: the last window is re-ranked first, then
  the window overlaps upward, and the best `stride` items of each window are
  fixed in place;
* unparsable output falls back to the input order for that window and is
  counted in the meta file, never silently repaired by scores.

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

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYSTEM = ("You are RankGPT, an intelligent assistant that can rank passages based "
          "on their relevancy to the query.")
INSTRUCTION = (
    "I will provide you with {n} passages, each indicated by a numeric identifier []. "
    "Rank the passages based on their relevance to the search query.\n"
    "Search Query: {q}\n\n{passages}\n\n"
    "Rank the {n} passages above based on their relevance to the search query. "
    "All passages should be included. Output only the identifiers of the passages "
    "in order of relevance, most relevant first, in the form: [1] > [2] > [3] ...")


def build_prompt(query, passages, start_idx):
    blocks = []
    for i, text in enumerate(passages):
        blocks.append(f"[{start_idx + i}] {text}")
    return INSTRUCTION.format(n=len(passages), q=query, passages="\n\n".join(blocks))


def parse_permutation(output, n):
    """Extract a permutation of 1..n from the model output; None if unusable."""
    nums = [int(x) for x in re.findall(r"\d+", output)]
    seen, perm = set(), []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            perm.append(x)
    if len(perm) != n:
        return None
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--passage-chars", type=int, default=200)
    ap.add_argument("--model", default=r"D:\RETECO\models\qwen")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to(device)

    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    queries = {}
    with open(os.path.join(ROOT, "data", "track1_tempo", args.domain,
                           f"examples_{args.split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["id"]] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ",
                                                          r["query"])).strip()

    cand = read_trec_run(args.run)
    qids = list(cand)
    if args.limit_queries:
        qids = qids[: args.limit_queries]

    results, lat, n_parse_fail, n_windows = {}, [], 0, 0
    for qi, qid in enumerate(qids):
        tq = time.time()
        docs = [t[2] for t in cand[qid][: args.k]]
        order = list(range(len(docs)))  # current order as indices into docs

        def short(i):
            return corpus.get(docs[i], "")[: args.passage_chars].replace("\n", " ")

        end = len(order)
        while end > 0:
            start = max(0, end - args.window)
            window_idx = order[start:end]
            passages = [short(i) for i in window_idx]
            prompt = build_prompt(queries[qid], passages, 1)
            chat = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            enc = tok(chat, return_tensors="pt", truncation=True, max_length=4096).to(device)
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            perm = parse_permutation(text, len(window_idx))
            n_windows += 1
            if perm is None:
                n_parse_fail += 1
            else:
                order[start:end] = [window_idx[p - 1] for p in perm]
            if start == 0:
                break
            end = start + args.stride

        results[qid] = [(docs[i], float(len(docs) - r)) for r, i in enumerate(order)]
        lat.append((time.time() - tq) * 1000.0)
        if (qi + 1) % 5 == 0:
            print(f"  {qi+1}/{len(qids)} queries", flush=True)

    write_trec_run(args.out, results, tag="listwise")
    meta = {
        "domain": args.domain, "split": args.split,
        "candidate_run": os.path.basename(args.run), "candidate_k": args.k,
        "window": args.window, "stride": args.stride,
        "passage_chars": args.passage_chars,
        "model": os.path.basename(args.model.rstrip("/\\")),
        "queries": len(qids), "windows": n_windows, "parse_failures": n_parse_fail,
        "parse_failure_rate": round(n_parse_fail / max(1, n_windows), 3),
        "seconds": round(time.time() - t0, 1),
        "latency_ms_per_query": {
            "mean": round(sum(lat) / len(lat), 1),
            "p50": round(sorted(lat)[len(lat) // 2], 1),
            "max": round(max(lat), 1),
        },
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device == "cuda" else None,
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["latency_ms_per_query"]))
    print(f"parse failures {n_parse_fail}/{n_windows}")


if __name__ == "__main__":
    main()
