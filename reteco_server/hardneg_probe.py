"""Sample hard negatives from the CE training groups and ask the judge whether
they actually answer the query (model-assisted estimate of false negatives)."""
import argparse, json, os, random, sys
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = "/homeC/wujiaheng/RETECO"
DOMAINS = sorted(d for d in os.listdir(f"{ROOT}/data/track1_tempo")
                 if os.path.exists(f"{ROOT}/data/track1_tempo/{d}/qrels_train.txt"))

def dense_rank(d):
    run = defaultdict(dict)
    for ln in open(f"{ROOT}/runs/dense_mxbai_{d}_train.trec", encoding="utf-8"):
        p = ln.split()
        if len(p) >= 4: run[p[0]][p[2]] = int(p[3])
    return run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/reteco/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="results/analysis/hardneg_probe.json")
    a = ap.parse_args()
    rng = random.Random(20261123)
    samples = []
    per_domain = max(1, a.n // len(DOMAINS))
    for d in DOMAINS:
        groups = [json.loads(l) for l in open(f"{ROOT}/results/ce_folds/fall/g_all_{d}.jsonl", encoding="utf-8") if l.strip()]
        rk = dense_rank(d)
        cand = []
        for g in groups:
            for doc in g["docs"]:
                if doc["label"] == 0 and rk.get(g["qid"], {}).get(doc["id"], 999) <= 20:
                    cand.append((g["qid"], g["query_full"], doc["id"], rk[g["qid"]][doc["id"]]))
        rng.shuffle(cand)
        samples += cand[:per_domain]
    corpus = {}
    by_dom = defaultdict(list)
    for s in samples: by_dom[s[0].split("/")[0] if "/" in s[0] else None]
    # docids look like "domain/xxx" -> map by prefix
    need = defaultdict(set)
    for qid, query, doc, rank in samples:
        need[doc.split("/")[0] if "/" in doc else "?"].add(doc)
    for d in DOMAINS:
        if not need.get(d): continue
        with open(f"{ROOT}/data/track1_tempo/{d}/documents.jsonl", encoding="utf-8") as fh:
            for line in fh:
                it = json.loads(line)
                if it["id"] in need[d]:
                    corpus[it["id"]] = it["content"]
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                trust_remote_code=True).to("cuda").eval()
    yes = tok.encode("yes")[0]; no = tok.encode("no")[0]
    out = []
    B = a.batch_size
    for i in range(0, len(samples), B):
        batch = samples[i:i + B]
        prompts = []
        for qid, query, doc, rank in batch:
            body = corpus.get(doc, "")[:4000]
            prompts.append(f"Question: {query[:1200]}\n\nPassage: {body}\n\n"
                           f"Does this passage contain evidence that answers the question? Answer yes or no.")
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
            p_yes = torch.softmax(logits[:, [yes, no]], dim=-1)[:, 0]
        for (qid, query, doc, rank), p in zip(batch, p_yes.tolist()):
            out.append({"qid": qid, "docid": doc, "dense_rank": rank, "p_yes": round(p, 4)})
        if i % (B * 10) == 0: print("scored", i, flush=True)
    os.makedirs(os.path.dirname(f"{ROOT}/{a.out}"), exist_ok=True)
    json.dump({"n": len(out), "thresholds": {t: sum(1 for o in out if o["p_yes"] >= t) for t in (0.5, 0.7, 0.9)},
               "records": out}, open(f"{ROOT}/{a.out}", "w"), indent=1)
    print("DONE", len(out), "p>=0.5:", sum(1 for o in out if o["p_yes"] >= 0.5))
main()
