"""Generate temporal sub-questions for each query with the local instruct model.

Output: {qid: [sub_question, ...]} JSON, plus raw generations for audit.

Prompt asks for 2-4 short sub-questions that cover the different time periods /
events / aspects required by the search query. No qrels are read; the model sees
only the query text (title + body).
"""
import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def clean(t):
    t = re.sub(r"<[^>]+>", " ", t or "")
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "qwen"))
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    qids, queries = [], []
    path = os.path.join(ROOT, "data", "track1_tempo", args.domain,
                        f"examples_{args.split}.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qids.append(r["id"])
            queries.append(clean(r["query"])[:1500])

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16).eval().cuda()

    prompt_t = (
        "A search query may require information about different time periods, "
        "events, or changes over time.\n"
        "Break the query into 2-4 short sub-questions, each targeting one time "
        "period, event, or aspect. Keep each sub-question under 20 words.\n"
        "Output one sub-question per line, starting with '- '. Do not explain.\n\n"
        "Query: {q}\n\nSub-questions:")

    out, raw = {}, {}
    for qid, q in zip(qids, queries):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt_t.format(q=q)}],
            tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt", truncation=True,
                  max_length=3072).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raw[qid] = text
        subs = []
        for line in text.splitlines():
            line = line.strip().lstrip("-*\u2022 ").strip()
            if 3 <= len(line.split()) <= 25 and line.endswith("?"):
                subs.append(line)
        out[qid] = subs[:4]
        print(f"{qid}: {len(out[qid])} sub-questions", flush=True)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(args.json.replace(".json", "_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
    print("wrote", args.json)


if __name__ == "__main__":
    main()
