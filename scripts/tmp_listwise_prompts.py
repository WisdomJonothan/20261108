"""Debug: test alternative listwise prompts on one query."""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = r"D:\RETECO"
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models", "qwen"))
model = AutoModelForCausalLM.from_pretrained(
    os.path.join(ROOT, "models", "qwen"), torch_dtype=torch.float16).eval().cuda()

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
        p = line.split()
        if p[0] == "2714_798" and int(p[3]) <= 20:
            cand.append(p[2])

passages = "\n\n".join(f"[{i+1}] {docs[d][:300]}".replace("\n", " ")
                       for i, d in enumerate(cand))


def ask(user):
    chat = tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True)
    enc = tok(chat, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


prompt_b = (f"Search query: {q}\n\n{passages}\n\n"
            "Which single passage above is most relevant to the search query? "
            "Answer with only its number in brackets, e.g. [7].")
print("B (top-1):", ask(prompt_b))

prompt_c = (f"Search query: {q}\n\n{passages}\n\n"
            "List the numbers of the three most relevant passages, most relevant "
            "first, in the form: [3], [11], [7].")
print("C (top-3):", ask(prompt_c))

prompt_d = (f"Search query: {q}\n\n{passages}\n\n"
            "Read each passage and compare it with the query. Then output the "
            "numbers 1 to 20 in the order of how well each passage answers the "
            "query, best first. Do not copy the original order.")
print("D (perm, explicit):", ask(prompt_d))
