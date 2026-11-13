#!/usr/bin/env python3
"""Score a candidate list with Qwen3-Reranker-0.6B (the modern baseline).

Reads the dense top-k the frozen system feeds its cross-encoder, scores every
(query, document) pair with the model's yes/no logit difference, and writes a
TREC run in the same shape as the in-house reranker so the rest of the
pipeline (`score_run.py`, `bootstrap_compare.py`, `dev_multievidence.py`)
works unchanged.

    /d/anaconda/python.exe scripts/run_qwen3_rerank.py --domain iota \
        --run runs/dense_mxbai_iota_dev.trec --model models/qwen3-reranker-0.6b \
        --out runs/rerank_qwen3_iota_dev.trec
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib_reteco import load_corpus, read_trec_run, write_trec_run  # noqa: E402


def clean_query(text):
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

SYSTEM = ("Judge whether the Document meets the requirements based on the Query "
          "and the Instruct provided. Note that the answer can only be "
          "\"yes\" or \"no\".")
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"


def load_queries(domain, split):
    path = os.path.join(ROOT, "data", "track1_tempo", domain,
                        "examples_%s.jsonl" % split)
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = clean_query(r["query"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="D:/RETECO/models/qwen3-reranker-0.6b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--doc-cap", type=int, default=6000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=dev).eval()
    yes = tok.convert_tokens_to_ids("yes")
    no = tok.convert_tokens_to_ids("no")

    queries = load_queries(args.domain, args.split)
    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    cand = read_trec_run(args.run)

    results = {}
    n_pairs = 0
    for qid in sorted(cand):
        q = queries.get(qid)
        if q is None:
            continue
        docs = [doc for _r, _s, doc in cand[qid]][:100]
        texts = [corpus[d][:args.doc_cap] for d in docs]
        pairs = []
        for t in texts:
            msg = [{"role": "system", "content": SYSTEM},
                   {"role": "user",
                    "content": "<Instruct>: %s\n<Query>: %s\n<Document>: %s"
                               % (INSTRUCT, q, t)}]
            pairs.append(tok.apply_chat_template(msg, tokenize=False,
                                                 add_generation_prompt=False))
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), args.batch):
                enc = tok(pairs[i:i + args.batch], padding=True, truncation=True,
                          max_length=args.max_len, return_tensors="pt").to(dev)
                logits = model(**enc).logits[:, -1, :]
                s = (logits[:, yes] - logits[:, no]).float().cpu().numpy()
                scores.extend(s.tolist())
        results[qid] = list(zip(docs, scores))
        n_pairs += len(docs)
        print(json.dumps({"domain": args.domain, "qid": qid, "docs": len(docs),
                          "scored": n_pairs}), flush=True)

    write_trec_run(args.out, results, tag="qwen3")
    print(json.dumps({"domain": args.domain, "queries": len(results),
                      "pairs": n_pairs, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
