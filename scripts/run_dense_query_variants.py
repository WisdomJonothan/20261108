"""Dense retrieval with alternative query representations.

Document embeddings are reused from the existing cache, so only the queries are
encoded: four query views over the same frozen corpus representation.

Views:
  full          — the whole query (HTML stripped)                     [baseline]
  title         — text before the first <p> (the question title)
  summary       — guidance.query_summary (organizer-written 1-2 sentences)
  title_summary — title + summary concatenated

The corpus encoding is identical to run_dense_bge.py (bge-base-en-v1.5, CLS
pooling, cosine, same instruction), so differences are attributable to the query
representation alone.

Usage:
  python run_dense_query_variants.py --domain bitcoin --split dev \
      --q-mode title --out runs/dense_title_bitcoin_dev.trec --meta results/...
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, write_trec_run  # noqa: E402

ROOT = r"D:\RETECO"
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def clean(t: str) -> str:
    t = re.sub(r"<[^>]+>", " ", t or "")
    t = (t.replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
         .replace("&nbsp;", " "))
    return re.sub(r"\s+", " ", t).strip()


def load_queries(domain, split, mode):
    data = os.path.join(ROOT, "data", "track1_tempo", domain)
    with open(os.path.join(data, f"examples_{split}.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    guidance = {}
    gpath = os.path.join(data, f"guidance_{split}.jsonl")
    if os.path.isfile(gpath):
        with open(gpath, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                guidance[r["id"]] = r["query_guidance"]

    qids, texts = [], []
    for r in recs:
        qid = r["id"]
        raw = clean(r["query"])
        title = clean(r["query"].split("<p>")[0]) or raw
        summary = clean(guidance.get(qid, {}).get("query_summary", ""))
        if mode == "full":
            text = raw
        elif mode == "title":
            text = title
        elif mode == "summary":
            text = summary or title
        elif mode == "title_summary":
            text = f"{title} {summary}".strip() if summary else title
        else:
            raise SystemExit(f"unknown mode {mode}")
        qids.append(qid)
        texts.append(text)
    return qids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--q-mode", required=True,
                    choices=["full", "title", "summary", "title_summary"])
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "bge-base-en-v1.5"))
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    ids, _texts = load_corpus(os.path.join(ROOT, "data", "track1_tempo", args.domain))
    cache = os.path.join(ROOT, "emb", f"doc_emb_bge-base-en-v1.5_{len(ids)}.npy")
    doc_emb = np.load(cache)

    qids, queries = load_queries(args.domain, args.split, args.q_mode)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).eval().cuda()
    q_emb = np.zeros((len(queries), doc_emb.shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(queries), 16):
            batch = [INSTRUCTION + q for q in queries[i:i + 16]]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state[:, 0].float()
            q_emb[i:i + 16] = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
    del model
    torch.cuda.empty_cache()

    doc_t = torch.from_numpy(doc_emb).cuda()
    ranked = {}
    for i, qid in enumerate(qids):
        sims = doc_t @ torch.from_numpy(q_emb[i]).cuda()
        k = min(args.topk, sims.shape[0])
        vals, idx = torch.topk(sims, k)
        ranked[qid] = [(ids[j], float(v)) for j, v in zip(idx.tolist(), vals.tolist())]
    write_trec_run(args.out, ranked, tag=f"dense-{args.q_mode}")

    meta = {"domain": args.domain, "split": args.split, "q_mode": args.q_mode,
            "queries": len(qids), "docs": len(ids), "topk": args.topk,
            "run_file": os.path.abspath(args.out),
            "doc_emb_cache": cache}
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{args.q_mode}] {len(qids)} queries over {len(ids)} docs -> {args.out}")


if __name__ == "__main__":
    main()
