"""Dense retrieval with Qwen3-Embedding-0.6B (last-token pooling, instruction queries).

Mirrors scripts/dense_mxbai.py but uses the Qwen3-Embedding instruction format:

    Instruct: Given a web search query, retrieve relevant passages that answer the query
    Query: <query text>

Documents are embedded as "title. text" with no instruction. Vectors are L2-normalised
and scored by dot product.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TASK = "Given a web search query, retrieve relevant passages that answer the query"
NL = chr(10)


def read_lines(path):
    # str.splitlines() also splits on U+2028/U+2029/U+0085, which occur inside some
    # corpus documents and would break one JSON object into two.
    return Path(path).read_text(encoding="utf-8").split(NL)


def load_docs(domain):
    path = ROOT / "data" / "track1_tempo" / domain / "documents.jsonl"
    ids, texts = [], []
    for line in read_lines(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        title = (o.get("title") or "").strip()
        body = (o.get("text") or o.get("content") or "").strip()
        ids.append(o.get("docid") or o.get("_id") or o.get("id"))
        texts.append(f"{title}. {body}" if title else body)
    return ids, texts


def load_queries(domain, split):
    path = ROOT / "data" / "track1_tempo" / domain / f"examples_{split}.jsonl"
    qids, texts = [], []
    for line in read_lines(path):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        qids.append(str(o.get("qid") or o.get("_id") or o.get("id")))
        texts.append(str(o.get("query") or o.get("text")))
    return qids, texts


@torch.no_grad()
def embed(model, tok, texts, batch, max_len, is_query, device="cuda"):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        if is_query:
            chunk = [f"Instruct: {TASK}" + NL + f"Query: {t}" for t in chunk]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        lengths = enc["attention_mask"].sum(dim=1) - 1
        vec = h[torch.arange(h.size(0)), lengths]
        vec = torch.nn.functional.normalize(vec.float(), dim=-1)
        out.append(vec.cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--domains", nargs="+", required=True)
    ap.add_argument("--tag", default="qwen3emb06")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--doc-max-len", type=int, default=512)
    ap.add_argument("--query-max-len", type=int, default=256)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--cache-dir", default="/tmp/reteco/emb")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.model, dtype=torch.float16).to("cuda").eval()

    cache = Path(a.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for dom in a.domains:
        t0 = time.time()
        ids, texts = load_docs(dom)
        out = ROOT / "runs" / f"dense_{a.tag}_{dom}_{a.split}.trec"
        if out.exists() and out.stat().st_size > 0:
            print(f"{dom}: already done -> {out.name}", flush=True)
            continue
        cpath = cache / f"{a.tag}_{dom}.npy"
        if cpath.exists():
            mat = np.load(cpath)
            print(f"{dom}: loaded cached embeddings {mat.shape}", flush=True)
        else:
            mat = embed(model, tok, texts, a.batch, a.doc_max_len, False)
            np.save(cpath, mat)
        qids, qtexts = load_queries(dom, a.split)
        qv = embed(model, tok, qtexts, a.batch, a.query_max_len, True)
        scores = qv @ mat.T
        with out.open("w", encoding="utf-8") as fh:
            for i, qid in enumerate(qids):
                k = min(a.topk, scores.shape[1])
                top = np.argpartition(-scores[i], k - 1)[:k]
                top = top[np.argsort(-scores[i][top])]
                for rank, j in enumerate(top, start=1):
                    fh.write(f"{qid} Q0 {ids[j]} {rank} {scores[i][j]:.6f} {a.tag}" + NL)
        print(f"{dom}: {len(ids)} docs {len(qids)} queries {time.time()-t0:.0f}s -> {out.name}",
              flush=True)
        cpath.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
