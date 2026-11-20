#!/usr/bin/env python3
"""Qwen3-Reranker controls for the bounded replacement study (step 4).

Every mode uses the frozen per-query 13-document pool (frozen top-10 + the <=3
frozen candidates from results/replacement_v2/candidates_dev.json), so all modes
are comparable with the Direct/Context arms and the oracle:

  single_swap : score the 13 docs, execute at most one (candidate -> rank 8/9/10)
                swap, the largest positive score gain
  full_rerank : sort the same 13 docs by reranker score, take the top 10
  deep100     : rerank the dense top-100 and take ten (a much larger budget)

Scores are the model's official yes/no logit comparison, so the same code path
serves Qwen3-Reranker-4B and any other Qwen3 reranker checkpoint.
"""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]
INSTRUCT = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
          "the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
          "<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n\n\n\n"


def read_qrels(path):
    gold = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            p = ln.split()
            if len(p) >= 4 and int(p[3]) > 0:
                gold.setdefault(p[0], set()).add(p[2])
    return gold


def ndcg10(ranked, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def read_trec_run(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            qid, _q0, docid, rank, score, _tag = ln.split()
            out.setdefault(qid, []).append((int(rank), float(score), docid))
    for qid in out:
        out[qid].sort(key=lambda r: r[0])
    return out


def truncate(text, max_words=220):
    return " ".join(str(text).split()[:max_words])


def load_corpus(directory, need):
    corpus = {}
    with (directory / "documents.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            if item["id"] in need:
                corpus[item["id"]] = item["content"]
                if len(corpus) == len(need):
                    break
    return corpus


@torch.inference_mode()
def score_docs(model, tokenizer, pairs, batch_size, max_length=1024):
    scores, tokens = [], 0
    yes_ids = [tokenizer.encode("yes", add_special_tokens=False)[0]]
    no_ids = [tokenizer.encode("no", add_special_tokens=False)[0]]
    assert len(yes_ids) == 1 and len(no_ids) == 1
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        texts = [PREFIX + f"<Instruct>: {INSTRUCT}\n<Query>: {q}\n<Document>: {d}" + SUFFIX
                 for q, d in chunk]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                        add_special_tokens=False, return_tensors="pt").to(model.device)
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        s = (lp[:, yes_ids[0]] - lp[:, no_ids[0]]).sigmoid().cpu().tolist()
        scores.extend(s)
        tokens += int(enc["attention_mask"].sum().item())
    return scores, tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--candidates", default="results/replacement_v2/candidates_dev.json")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--modes", default="single_swap,full_rerank")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--domains", default="")
    ap.add_argument("--out", default="results/replacement_v2/reranker_controls_dev.json")
    ap.add_argument("--runs-dir", default="results/replacement_v2/reranker_runs")
    args = ap.parse_args()
    root = args.root
    domains = args.domains.split(",") if args.domains else DOMAINS
    payload = json.loads((root / args.candidates).read_text(encoding="utf-8"))
    by_domain = {d: [r for r in payload["queries"] if r["domain"] == d] for d in domains}

    path = args.model or root / "models/Qwen3-Reranker-4B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    # Qwen tokenizers default to right padding; the yes/no logits are read at the
    # last position, so a right-padded batch would score pad tokens. Left padding
    # reproduces the unpadded per-document reference (verified: Spearman 0.998,
    # top-1 agreement 96% at batch 8; right padding gives 0.43 / 27%).
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True,
        dtype=torch.float16 if device == "cuda" else torch.float32).eval().to(device)

    result = {"model": str(path), "candidate_file": args.candidates, "instruct": INSTRUCT,
              "n_queries": sum(len(v) for v in by_domain.values()), "modes": {}}
    runs_dir = root / args.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes.split(","):
        per_query = []
        docs_scored = tokens_total = 0
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        for domain in domains:
            directory = root / "data/track1_tempo" / domain
            gold = read_qrels(directory / "qrels_dev.txt")
            rows = by_domain[domain]
            if args.limit_queries:
                rows = rows[: args.limit_queries]
            need = set()
            for row in rows:
                need |= set(row["frozen_top10"]) | {c["docid"] for c in row["candidates"]}
            dense = read_trec_run(root / "runs" / f"dense_mxbai_{domain}_dev.trec")
            if mode == "deep100":
                for row in rows:
                    need |= {d for _, _, d in dense[row["qid"]][:100]}
            if mode == "start20":
                for row in rows:
                    need |= {d for _, _, d in dense[row["qid"]][:20]}
            corpus = load_corpus(directory, need)
            for row in rows:
                top = row["frozen_top10"]
                g = gold.get(row["qid"], set())
                base = ndcg10(top, g)
                if mode == "deep100":
                    cand100 = [d for _, _, d in dense[row["qid"]][:100] if d in corpus]
                    docs = cand100
                elif mode == "start20":
                    docs = [d for _, _, d in dense[row["qid"]][:20] if d in corpus]
                else:
                    docs = list(dict.fromkeys(top + [c["docid"] for c in row["candidates"]]))
                scores, ntokens = score_docs(
                    model, tokenizer, [(row["query"], truncate(corpus[d])) for d in docs],
                    args.batch_size)
                docs_scored += len(docs)
                tokens_total += ntokens
                s = dict(zip(docs, scores))
                if mode in ("deep100", "start20"):
                    ranking = sorted(docs, key=lambda d: -s[d])[:10]
                    swap = None
                    if mode == "start20":
                        base = ndcg10(ranking, g)
                elif mode == "full_rerank":
                    ranking = sorted(docs, key=lambda d: -s[d])[:10]
                    swap = {"type": "full_rerank", "pool": len(docs)}
                else:
                    best = None
                    for c in row["candidates"]:
                        for slot in (7, 8, 9):
                            gain = s[c["docid"]] - s[top[slot]]
                            if gain > 0 and (best is None or gain > best[0]):
                                best = (gain, slot, c["docid"], top[slot])
                    ranking = list(top)
                    swap = None
                    if best is not None:
                        ranking[best[1]] = best[2]
                        swap = {"slot_rank": best[1] + 1, "removed": best[3],
                                "incoming": best[2], "score_gain": best[0]}
                if mode == "deep100":
                    keep = sorted(docs, key=lambda d: -s[d])[:20]
                    shown = {d: s[d] for d in keep}
                else:
                    shown = dict(s)
                per_query.append({"domain": domain, "qid": row["qid"], "baseline": base,
                                  "policy": ndcg10(ranking, g), "swap": swap, "ranking": ranking,
                                  "n_scored": len(docs), "scores": shown})
        elapsed = time.perf_counter() - t_start
        for r in per_query:
            r["delta"] = r["policy"] - r["baseline"]
        macro_base = float(np.mean([np.mean([r["baseline"] for r in per_query if r["domain"] == d])
                                    for d in domains]))
        macro = float(np.mean([np.mean([r["policy"] for r in per_query if r["domain"] == d])
                               for d in domains]))
        counts = Counter("no_swap" if r["swap"] is None else
                         "helpful" if r["delta"] > 1e-12 else
                         "harmful" if r["delta"] < -1e-12 else "neutral" for r in per_query)
        rng = np.random.default_rng(20261120)
        deltas = {d: np.array([r["delta"] for r in per_query if r["domain"] == d]) for d in domains}
        sims = np.array([np.mean([v[rng.integers(len(v), size=len(v))].mean()
                                  for v in deltas.values()]) for _ in range(10000)])
        result["modes"][mode] = {
            "macro_baseline": macro_base, "macro_policy": macro, "macro_delta": macro - macro_base,
            "swap_counts": dict(counts),
            "ci95": [float(np.quantile(sims, .025)), float(np.quantile(sims, .975))],
            "cost": {"docs_scored": docs_scored, "prompt_tokens": tokens_total,
                     "seconds": elapsed, "docs_per_s": docs_scored / max(elapsed, 1e-9),
                     "gpu_peak_mb": (torch.cuda.max_memory_allocated() / 2 ** 20)
                     if device.startswith("cuda") else 0.0},
        }
        for domain in domains:
            lines = []
            for r in per_query:
                if r["domain"] != domain:
                    continue
                for i, doc in enumerate(r["ranking"], 1):
                    lines.append(f"{r['qid']}\tQ0\t{doc}\t{i}\t{1.0 / i:.6f}\t{mode}")
            if lines:
                (runs_dir / f"{mode}_{domain}_dev.trec").write_text("\n".join(lines) + "\n",
                                                                    encoding="utf-8")
        print(mode, json.dumps(result["modes"][mode]), flush=True)
        result.setdefault("per_query", {})[mode] = [
            {k: r[k] for k in ("domain", "qid", "baseline", "policy", "delta", "swap")}
            for r in per_query]

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
