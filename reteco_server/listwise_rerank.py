"""Listwise rerank of the frozen top-20 with Qwen3-4B-Instruct-2507.

For every dev query: take the frozen top-20 (sub_sft_e2bc), number the passages,
ask the model for the 10 most relevant ids in order, parse, reorder, score.

This tests the "capability at the head" hypothesis: the tail-swap protocol only
looked at ranks 8-10, where 1/log2(1+r) is smallest.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_pairwise_replacement_v2 as B  # noqa: E402

DOMAINS13 = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
             "law", "monero", "politics", "quant", "travel", "workplace"]

PROMPT = """You are ranking passages for a search question. The question states conditions
that a useful passage must satisfy: a time period or version, a named entity or jurisdiction,
numeric bounds, or exclusions.

Question:
{query}

Candidates:
{cands}

Output the ids of the 10 most relevant passages, best first, as a comma-separated list of
numbers only (for example: 4, 12, 1, 7, 3, 9, 2, 15, 6, 11).
"""


def load_docs(domain):
    docs = {}
    p = Path("data/track1_tempo") / domain / "documents.jsonl"
    with p.open(encoding="utf-8") as fh:
        for ln in fh:
            o = json.loads(ln)
            docs[o["id"]] = o.get("content") or (o.get("title", "") + ". " + o.get("text", ""))
    return docs


def read_run(path):
    out = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 6:
            # p[3] is the RANK column, p[4] the score; sort by score then rank
            out.setdefault(p[0], []).append((p[2], float(p[4]), int(p[3])))
    for q in out:
        out[q].sort(key=lambda x: (-x[1], x[2]))
    return {q: [(doc, sc) for doc, sc, _ in v] for q, v in out.items()}
    return out


def build_prompt(query, docs):
    blocks = []
    for i, text in enumerate(docs, 1):
        snippet = " ".join(text.split())[:900]
        blocks.append(f"[{i}] {snippet}")
    return PROMPT.format(query=query.strip(), cands="\n".join(blocks))


def parse_ids(text, n):
    ids = []
    for m in re.finditer(r"\d+", text):
        k = int(m.group())
        if 1 <= k <= n and k not in ids:
            ids.append(k)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--domains", default=",".join(DOMAINS13))
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--max-words", type=int, default=90)
    ap.add_argument("--out", default="results/replacement_v2/listwise20_dev.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    domains = [d for d in args.domains.split(",") if d]
    per_query = []
    t0 = time.time()
    n_prompts = 0
    n_tokens = 0
    for d in domains:
        run = read_run(f"runs/sub_sft_e2bc_{d}_dev.trec")
        gold = B.read_qrels(f"data/track1_tempo/{d}/qrels_dev.txt")
        docs = load_docs(d)
        qtext = {}
        with open(f"data/track1_tempo/{d}/examples_dev.jsonl", encoding="utf-8") as fh:
            for ln in fh:
                o = json.loads(ln)
                qtext[o["id"]] = o["query"]
        qids = [q for q in gold if q in run]
        if args.limit_queries:
            qids = qids[: args.limit_queries]
        prompts = []
        meta = []
        for q in qids:
            top = [doc for doc, _ in run[q][: args.depth]]
            texts = []
            for doc in top:
                t = docs.get(doc, "")
                words = t.split()[: args.max_words]
                texts.append(" ".join(words))
            prompts.append(build_prompt(qtext.get(q, ""), texts))
            meta.append((q, top))
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            chat = [tok.apply_chat_template([{"role": "user", "content": b}], tokenize=False,
                                            add_generation_prompt=True) for b in batch]
            enc = tok(chat, return_tensors="pt", padding=True).to("cuda")
            n_prompts += len(batch)
            n_tokens += int(enc["input_ids"].numel())
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=48, do_sample=False)
            gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, text in enumerate(gen):
                q, top = meta[i + j]
                ids = parse_ids(text, len(top))
                ranked = [top[k - 1] for k in ids][:10]
                if len(ranked) < 10:
                    ranked += [doc for doc in top if doc not in ranked][: 10 - len(ranked)]
                per_query.append({"domain": d, "qid": q, "frozen": top[:10], "ranked": ranked,
                                  "delta": B.ndcg10(ranked, gold[q]) - B.ndcg10(top[:10], gold[q]),
                                  "raw": text.strip()[:80]})
        print(f"[{d}] done, {len(qids)} queries, {time.time()-t0:.0f}s", flush=True)

    per_dom = {}
    for r in per_query:
        per_dom.setdefault(r["domain"], []).append(r["delta"])
    macro_delta = sum(sum(v) / len(v) for v in per_dom.values()) / len(per_dom)
    base = {}
    for d in domains:
        gold = B.read_qrels(f"data/track1_tempo/{d}/qrels_dev.txt")
        run = read_run(f"runs/sub_sft_e2bc_{d}_dev.trec")
        base[d] = sum(B.ndcg10([x for x, _ in run[q][:10]], gold[q]) for q in gold if q in run) / len(gold)
    baseline_macro = sum(base.values()) / len(base)
    pol = {}
    for d in domains:
        gold = B.read_qrels(f"data/track1_tempo/{d}/qrels_dev.txt")
        run = read_run(f"runs/sub_sft_e2bc_{d}_dev.trec")
        vals = []
        for r in per_query:
            if r["domain"] != d:
                continue
            vals.append(B.ndcg10(r["ranked"], gold[r["qid"]]))
        pol[d] = sum(vals) / len(vals) if vals else 0.0
    macro = sum(pol.values()) / len(pol)
    res = {"model": args.model, "depth": args.depth, "max_words": args.max_words,
           "n_queries": len(per_query), "baseline_macro": baseline_macro,
           "macro_policy": macro, "macro_delta": macro - baseline_macro,
           "per_domain": {d: {"baseline": base[d], "policy": pol[d]} for d in domains},
           "cost": {"prompts": n_prompts, "prompt_tokens": n_tokens, "seconds": time.time() - t0},
           "per_query": per_query}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"baseline {baseline_macro:.6f} -> listwise {macro:.6f} (delta {macro-baseline_macro:+.6f})")
    print(f"prompts {n_prompts}, tokens {n_tokens}, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
