"""Manual two-model comparison on the same (query, document) pairs."""
import json
import os
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = r"D:\RETECO"
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
q = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", q)).strip()

ids = ["bitcoin/8e7db0cd_18300.txt", "bitcoin/8d3bb78c_18302.txt",
       "bitcoin/684017f8_143082.txt", "bitcoin/eadbcf09_7507.txt"]
names = ["gold_rpc", "gold_fmt", "top1_generic", "top2_timelock"]

for name, mdir in [("minilm", os.path.join(ROOT, "models", "ms-marco-MiniLM-L-6-v2")),
                   ("bge", os.path.join(ROOT, "models", "bge-reranker-base"))]:
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir).cuda().eval()
    with torch.no_grad():
        enc = tok([q] * len(ids), [docs[i] for i in ids], padding=True,
                  truncation=True, max_length=512, return_tensors="pt").to("cuda")
        sc = model(**enc).logits.view(-1).float().cpu().tolist()
    print(name, {n: round(s, 2) for n, s in zip(names, sc)})
    del model
    torch.cuda.empty_cache()
