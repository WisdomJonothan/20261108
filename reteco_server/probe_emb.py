import json, sys, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import importlib.util
spec = importlib.util.spec_from_file_location("dq", "scripts/dense_qwen3.py")
dq = importlib.util.module_from_spec(spec); spec.loader.exec_module(dq)
tok = AutoTokenizer.from_pretrained("/homeC/wujiaheng/models-qwen3-emb06", padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained("/homeC/wujiaheng/models-qwen3-emb06", dtype=torch.float16).to("cuda").eval()
dom = "bitcoin"
ids, texts = dq.load_docs(dom)
qids, qtexts = dq.load_queries(dom, "dev")
gold = {}
for line in open(f"data/track1_tempo/{dom}/qrels_dev.txt", encoding="utf-8"):
    p = line.split()
    gold.setdefault(p[0], set()).add(p[2])
q = qids[0]
gy = [d for d in gold[q] if d in set(ids)]
print("query:", qtexts[0][:180].replace("\n"," "))
print("relevant in corpus:", len(gy))
qv = dq.embed(model, tok, [qtexts[0]], 1, 256, True)[0]
# encode only a few docs: 2 relevant + 5 random
idx = [ids.index(d) for d in gy[:2]] + list(range(5))
sub = [texts[i] for i in idx]
dv = dq.embed(model, tok, sub, 7, 512, False)
for j, i in enumerate(idx):
    print(f"{'GOLD' if j < 2 else 'rand'} {ids[i][:40]:42s} cos={float(dv[j] @ qv):.4f}")
# also raw query without instruction
qv2 = dq.embed(model, tok, ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" + qtexts[0]], 1, 256, True)[0]
print("after-instruction cos(it):", float(dv[0] @ qv2))
