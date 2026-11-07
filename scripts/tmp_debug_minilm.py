"""Debug: replicate run_chunk_rerank's MiniLM path for one query."""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"D:\RETECO"
md = os.path.join(ROOT, "models", "ms-marco-MiniLM-L-6-v2")
tok = AutoTokenizer.from_pretrained(md)
model = AutoModelForSequenceClassification.from_pretrained(md).eval().cuda()

docs = {}
with open(os.path.join(ROOT, "data", "track1_tempo", "bitcoin", "documents.jsonl"),
          encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        docs[r["id"]] = r["content"]

q = None
with open(os.path.join(ROOT, "data", "track1_tempo", "bitcoin", "examples_dev.jsonl"),
          encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] == "97135_89":
            q = r["query"]

q_ids = tok(q, add_special_tokens=False, truncation=True, max_length=128)["input_ids"]
q_text = tok.decode(q_ids)
print("query text:", repr(q_text[:80]), "ids", len(q_ids))


def windows_of(text, win=380, stride=380):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= win:
        return [tok.decode(ids)] if ids else [""]
    out = []
    for start in range(0, len(ids), stride):
        chunk = ids[start:start + win]
        if not chunk:
            break
        out.append(tok.decode(chunk))
        if start + win >= len(ids):
            break
    return out


ids4 = ["bitcoin/8e7db0cd_18300.txt", "bitcoin/8d3bb78c_18302.txt",
        "bitcoin/684017f8_143082.txt", "bitcoin/eadbcf09_7507.txt"]
with torch.inference_mode():
    for d in ids4:
        ws = windows_of(docs[d])
        enc = tok([q_text] * len(ws), ws, padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to("cuda")
        logits = model(**enc).logits
        print(d, "windows", len(ws), "logits shape", tuple(logits.shape),
              "scores", logits.view(-1).float().cpu().numpy().round(3).tolist(),
              "lens", enc["attention_mask"].sum(1).tolist())
