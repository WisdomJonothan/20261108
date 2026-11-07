"""Debug: print the raw listwise model output for one query."""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = r"D:\RETECO"
md = os.path.join(ROOT, "models", "qwen")
tok = AutoTokenizer.from_pretrained(md)
model = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float16).eval().cuda()

docs = {}
with open(os.path.join(ROOT, "data", "track1_tempo", "iota", "documents.jsonl"),
          encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        docs[r["id"]] = r["content"]

q = None
with open(os.path.join(ROOT, "data", "track1_tempo", "iota", "examples_dev.jsonl"),
          encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] == "2714_798":
            q = r["query"]

cand = []
with open(os.path.join(ROOT, "runs", "dense_bge_iota_dev.trec"), encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if parts[0] == "2714_798" and int(parts[3]) <= 20:
            cand.append(parts[2])

passages = "\n\n".join(f"[{i+1}] {docs[d][:200]}".replace("\n", " ")
                       for i, d in enumerate(cand))
system = ("You are RankGPT, an intelligent assistant that can rank passages based "
          "on their relevancy to the query.")
user = (
    f"I will provide you with {len(cand)} passages, each indicated by a numeric "
    f"identifier []. Rank the passages based on their relevance to the search query.\n"
    f"Search Query: {q}\n\n{passages}\n\n"
    f"Rank the {len(cand)} passages above based on their relevance to the search "
    f"query. All passages should be included. Output only the identifiers of the "
    f"passages in order of relevance, most relevant first, in the form: "
    f"[1] > [2] > [3] ...")
chat = tok.apply_chat_template([{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                               tokenize=False, add_generation_prompt=True)
enc = tok(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.eos_token_id)
print("=== OUTPUT ===")
print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
