"""Probe: does the local instruct model have any usable relevance signal?

For each query, take one gold document and one random non-gold document from the
dense top-100, present them in BOTH orders, and ask which is more relevant.
A model with signal should answer the gold passage (regardless of position) well
above chance; a model that just echoes position will score ~50% and be
position-biased (always picking the first).

Usage: python probe_pairwise.py --domain bitcoin --split dev --n 40
"""
import argparse
import json
import os
import random
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_qrels, read_trec_run  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bitcoin")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--run", default=None)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "qwen"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--chars", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    run = args.run or os.path.join(
        ROOT, "runs", f"dense_bge_{args.domain}_{args.split}.trec")
    qrels = load_qrels(os.path.join(ROOT, "data", "track1_tempo", args.domain),
                       args.split)
    ids, texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    corpus = dict(zip(ids, texts))
    cand = read_trec_run(run)
    queries = {}
    with open(os.path.join(ROOT, "data", "track1_tempo", args.domain,
                           f"examples_{args.split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["id"]] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ",
                                                          r["query"])).strip()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).eval().cuda()

    def ask(prompt):
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt", truncation=True,
                  max_length=4096).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    rows = []
    for qid in list(cand)[: args.n]:
        docs = [t[2] for t in cand[qid]]
        gold = [d for d in docs if d in qrels.get(qid, {})]
        nongold = [d for d in docs if d not in qrels.get(qid, {})]
        if not gold or not nongold:
            continue
        g = random.choice(gold)
        ng = random.choice(nongold)
        for order in ("gold_first", "nongold_first"):
            a, b = (g, ng) if order == "gold_first" else (ng, g)
            prompt = (f"Search query: {queries[qid]}\n\n"
                      f"[A] {corpus.get(a,'')[:args.chars]}\n\n"
                      f"[B] {corpus.get(b,'')[:args.chars]}\n\n"
                      "Which passage is more relevant to the search query? "
                      "Answer with A or B only.")
            ans = ask(prompt).strip()
            pick = "A" if re.search(r"\bA\b", ans[:10]) else ("B" if re.search(r"\bB\b", ans[:10]) else "?")
            correct = (pick == "A" and order == "gold_first") or (
                pick == "B" and order == "nongold_first")
            rows.append({"qid": qid, "order": order, "answer": ans[:20],
                         "pick": pick, "correct": bool(correct)})

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else None
    first = sum(1 for r in rows if r["pick"] == "A") / n if n else None
    by_order = {}
    for order in ("gold_first", "nongold_first"):
        sub = [r for r in rows if r["order"] == order]
        by_order[order] = sum(r["correct"] for r in sub) / len(sub) if sub else None
    out = {"domain": args.domain, "split": args.split, "model": os.path.basename(args.model),
           "pairs": n, "accuracy": acc, "always_pick_A_rate": first,
           "accuracy_by_order": by_order, "rows": rows}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
