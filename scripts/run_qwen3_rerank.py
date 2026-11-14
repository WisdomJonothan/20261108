#!/usr/bin/env python3
"""Zero-shot baseline: Qwen3-Reranker-0.6B on the frozen candidate list.

Scores every (query, document) pair the frozen cross-encoder would see (dense
top-100) with the model's yes/no probability, following the model card:

  prefix = <|im_start|>system ... <|im_end|>\n<|im_start|>user\n
  body   = <Instruct>: ...\n<Query>: ...\n<Document>: ...
  suffix = <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n

The body is tokenised with `max_length - len(prefix) - len(suffix)`, so the
truncation can never eat the assistant suffix; the tokens are then assembled
prefix + body + suffix. The score is P("yes") over the [no, yes] pair at the
last position (monotone with the logit difference).

Writes a TREC run sorted by score, consumable by fuse_scores.py / score_run.py.
"""
import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib_reteco import load_corpus, read_trec_run, write_trec_run  # noqa: E402

SYSTEM = ("Judge whether the Document meets the requirements based on the Query "
          "and the Instruct provided. Note that the answer can only be "
          "\"yes\" or \"no\".")
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n" % SYSTEM)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def clean_query(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


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
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--qids", default="", help="comma-separated qids to score (default: all)")
    ap.add_argument("--doc-cap", type=int, default=6000)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=dev).eval()
    yes = tok.convert_tokens_to_ids("yes")
    no = tok.convert_tokens_to_ids("no")
    prefix_tokens = tok.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tok.encode(SUFFIX, add_special_tokens=False)
    body_budget = args.max_len - len(prefix_tokens) - len(suffix_tokens)
    if body_budget < 128:
        raise SystemExit("max-len too small for the prompt scaffolding")

    queries = load_queries(args.domain, args.split)
    if args.qids:
        keep = {x.strip() for x in args.qids.split(",") if x.strip()}
        queries = {k: v for k, v in queries.items() if k in keep}
        if not queries:
            raise SystemExit("no requested qids found in this split")
    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    cand = read_trec_run(args.run)

    results, n_pairs, n_cut = {}, 0, 0
    for qid in sorted(cand):
        q = queries.get(qid)
        if q is None:
            continue
        docs = [doc for _r, _s, doc in cand[qid]][:100]
        bodies = ["<Instruct>: %s\n<Query>: %s\n<Document>: %s" % (
            INSTRUCT, q, corpus[d][:args.doc_cap]) for d in docs]
        enc = tok(bodies, padding=False, truncation="longest_first",
                  return_attention_mask=False, max_length=body_budget)
        rows = [prefix_tokens + e + suffix_tokens for e in enc["input_ids"]]
        n_cut += sum(1 for e in enc["input_ids"] if len(e) >= body_budget)
        bs = args.batch
        while True:
            scores = []
            try:
                with torch.no_grad():
                    for i in range(0, len(rows), bs):
                        batch = tok.pad({"input_ids": rows[i:i + bs]},
                                        padding=True, return_tensors="pt")
                        out = model(input_ids=batch["input_ids"].to(dev),
                                    attention_mask=batch["attention_mask"].to(dev),
                                    logits_to_keep=1).logits[:, -1, :].float()
                        pair = torch.stack([out[:, no], out[:, yes]], dim=1)
                        scores.extend(torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist())
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(json.dumps({"oom": True, "qid": qid, "retry_batch": bs}), flush=True)
        order = sorted(range(len(docs)), key=lambda i: -scores[i])
        results[qid] = [(docs[i], scores[i]) for i in order]
        torch.cuda.empty_cache()
        n_pairs += len(docs)
        print(json.dumps({"domain": args.domain, "qid": qid, "docs": len(docs),
                          "scored": n_pairs}), flush=True)

    write_trec_run(args.out, results, tag="qwen3")
    meta = {"domain": args.domain, "queries": len(results), "pairs": n_pairs,
            "max_len": args.max_len, "body_budget": body_budget,
            "doc_cap": args.doc_cap, "truncated_bodies": n_cut,
            "prefix_tokens": len(prefix_tokens), "suffix_tokens": len(suffix_tokens),
            "protocol": "Qwen3 model card: prefix + truncated body + think suffix, P(yes)"}
    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
