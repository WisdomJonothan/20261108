"""HyDE-style query expansion: retrieve with a hypothetical answer passage.

For each query the local instruct model writes a short hypothetical passage that
would answer it; that passage is embedded and used as the query vector
(optionally averaged with the original query embedding).

This tests whether *generative* query representation helps the first stage,
which is a different lever from the temporal features tried before.

Usage:
  python hyde_retrieval.py --domain bitcoin --split dev --alpha 0.5 \
      --out runs/hyde_bitcoin_dev.trec --meta results/hyde_bitcoin_dev.json
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


def clean(t):
    t = re.sub(r"<[^>]+>", " ", t or "")
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight of the HyDE vector vs the original query vector")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "qwen"))
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    data = os.path.join(ROOT, "data", "track1_tempo", args.domain)
    qids, queries = [], []
    with open(os.path.join(data, f"examples_{args.split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qids.append(r["id"])
            queries.append(clean(r["query"])[:1200])

    # 1) generate hypothetical passages
    lt = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).eval().cuda()
    prompt_t = ("Write a short passage (3-4 sentences) that would answer this search "
                "query. Do not add any preamble.\n\nQuery: {q}\n\nPassage:")
    hyde = []
    for q in queries:
        chat = lt.apply_chat_template([{"role": "user", "content": prompt_t.format(q=q)}],
                                      tokenize=False, add_generation_prompt=True)
        enc = lt(chat, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.inference_mode():
            gen = lm.generate(**enc, max_new_tokens=120, do_sample=False,
                              pad_token_id=lt.eos_token_id)
        hyde.append(lt.decode(gen[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip())
    del lm
    torch.cuda.empty_cache()

    # 2) encode queries and HyDE passages with the bi-encoder
    et = AutoTokenizer.from_pretrained(os.path.join(ROOT, "models", "bge-base-en-v1.5"))
    em = AutoModel.from_pretrained(os.path.join(ROOT, "models", "bge-base-en-v1.5"),
                                   torch_dtype=torch.float16).eval().cuda()

    def enc(texts, prefix):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), 16):
                b = [prefix + t for t in texts[i:i + 16]]
                e = et(b, padding=True, truncation=True, max_length=512,
                       return_tensors="pt").to("cuda")
                h = em(**e).last_hidden_state[:, 0].float()
                out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out, 0)

    q_emb = enc(queries, INSTRUCTION)
    h_emb = enc(hyde, "")
    del em
    torch.cuda.empty_cache()

    ids, _ = load_corpus(data)
    doc_emb = np.load(os.path.join(ROOT, "emb",
                                   f"doc_emb_bge-base-en-v1.5_{len(ids)}.npy"))
    doc_t = torch.from_numpy(doc_emb).cuda()
    ranked = {}
    for i, qid in enumerate(qids):
        v = args.alpha * h_emb[i] + (1 - args.alpha) * q_emb[i]
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = doc_t @ torch.from_numpy(v).cuda()
        k = min(args.topk, sims.shape[0])
        vals, idx = torch.topk(sims, k)
        ranked[qid] = [(ids[j], float(x)) for j, x in zip(idx.tolist(), vals.tolist())]
    write_trec_run(args.out, ranked, tag="hyde")

    meta = {"domain": args.domain, "split": args.split, "alpha": args.alpha,
            "queries": len(qids), "run_file": os.path.abspath(args.out),
            "hyde_samples": hyde[:3]}
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"hyde {args.domain}/{args.split}: {len(qids)} queries, alpha={args.alpha}")


if __name__ == "__main__":
    main()
