"""Diver-GroupRank-7B control: group-wise rerank of the delivered dense top-100.

Official template (model card): a system prompt describing usefulness-then-
relevance evaluation, then one user message containing up to TOPK documents
numbered [1]..[K] asking for scores in a JSON block inside <answer>.

Deviations from the card are recorded in the payload: greedy decoding instead of
temperature-0.3 sampling (reproducibility), groups of --group documents instead
of the whole 100 in one prompt, and a per-document word cap so a 100-document
request fits a 24 GB card.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import lib_reteco as R  # noqa: E402
from ir_metrics import ndcg_at_k  # noqa: E402

CH_NL = chr(10)
def fmt_passages(chunk, corpus, words):
    parts = []
    for i, d in enumerate(chunk):
        parts.append("[" + str(i + 1) + "]. " + " ".join(corpus.get(d, "").split()[:words]))
    return (CH_NL + CH_NL).join(parts)

def model_scores(model, tok, prompt, max_new_tokens):
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=16384).to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

DATA = Path("/homeC/wujiaheng/RETECO/data")
RUNS = Path("/homeC/wujiaheng/RETECO/runs")

SYS_PROMPT = """Your task is to evaluate and rank documents based on how well they help answer the given query. Follow this evaluation priority:
1. PRIMARY: Usefulness & Helpfulness - Does the document provide actionable information, solutions, or direct answers that help address the user's needs?
2. SECONDARY: Relevance - Does the document contain information related to the query topic?

Evaluation Process:
1. First, identify the user's core intent and what kind of help they need from the query
2. For each document, assess:
   - How directly it addresses the user's intent
   - What actionable information or answers it provides
   - How much it helps solve the user's problem or need
3. Compare documents against each other to ensure proper ranking
4. Assign scores that reflect the relative usefulness ranking

Scoring Scale (0-10):
- 9-10: Extremely helpful, directly answers the query with actionable information
- 7-8: Very helpful, provides substantial useful information for the query
- 5-6: Moderately helpful, contains some useful information but incomplete
- 3-4: Minimally helpful, limited useful information despite topic relevance
- 1-2: Barely helpful, mentions related topics but provides little useful information
- 0: Not helpful at all, cannot assist with answering the query
"""

USER_PROMPT = """I will provide you {TOPK} documents, each indicated by a numerical identifier []. Score these documents based on their Usefulness and Relevance to the query.
Query:
{QUERY}

Documents:
{PASSAGES}

## Final Output Format
You must structure your response in exactly two parts: provide your brief reasoning process first, then output final scores in JSON format like below, with document IDs as string keys and integer scores as values for all {TOPK} documents.
The reasoning process and answer are enclosed within <reason> </reason> and <answer> </answer> tags, respectively. Do NOT output anything outside the specified tags. Follow this exact format:
<reason>
[Analyze each document's usefulness and relevance to the query, explaining your scoring rationale]
</reason>
<answer>
```json
{{"[1]": 5, "[2]": 3, "[3]": 8}}
```
</answer>
"""


def parse_scores(text):
    """Pull "[i]": v pairs out of the <answer> block; return {index: score}."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.S)
    body = m.group(1) if m else text
    out = {}
    for idx, val in re.findall(r'"?\[(\d+)\]"?\s*:\s*(-?\d+(?:\.\d+)?)', body):
        out[int(idx)] = float(val)
    return out


def read_trec(path):
    runs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 5:
                runs[p[0]].append((p[2], float(p[4])))
    for q in runs:
        runs[q].sort(key=lambda x: -x[1])
    return runs


def read_qrels(path):
    gold = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 4 and int(p[3]) > 0:
                gold.setdefault(p[0], set()).add(p[2])
            elif len(p) == 3 and int(p[2]) > 0:
                gold.setdefault(p[0], set()).add(p[1])
    return gold


def clean_query(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = text.replace("&quot;", '"').replace("&amp;", "&").replace("&#39;", "'")
    return re.sub(r"\s+", " ", text).strip()


def query_text(raw):
    if "<p>" in raw:
        title, body = raw.split("<p>", 1)
    else:
        title, body = raw, ""
    return (clean_query(title) + " " + clean_query(body)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/homeC/wujiaheng/models-diver7b")
    ap.add_argument("--split", default="train")
    ap.add_argument("--qids-file", default="")
    ap.add_argument("--domains", default="")
    ap.add_argument("--per-domain", type=int, default=3)
    ap.add_argument("--group", type=int, default=20)
    ap.add_argument("--doc-words", type=int, default=220)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--pool", type=int, default=100, help="documents considered from the dense run")
    ap.add_argument("--cascade", action="store_true",
                    help="two rounds: keep the best --cascade-keep per group and rerank the survivors")
    ap.add_argument("--cascade-keep", type=int, default=5)

    ap.add_argument("--zscore-groups", action="store_true",
                    help="normalise scores inside each group before merging (groups are scored independently)")
    ap.add_argument("--prefill", action="store_true",
                    help="append the answer header so the model emits scores directly instead of reasoning first")
    ap.add_argument("--stop-strings", default="```")
    ap.add_argument("--out", default="results/analysis/diver_train.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).eval().to("cuda")
    base_dir = DATA / "track1_tempo"
    domains = a.domains.split(",") if a.domains else sorted(p.name for p in base_dir.iterdir())

    qids_filter = set()
    if a.qids_file:
        qids_filter = set(json.loads(Path(a.qids_file).read_text())["calibration"])

    per_query, t0, n_prompts = [], time.time(), 0
    for dom in domains:
        ddir = base_dir / dom
        ex_path = ddir / ("examples_%s.jsonl" % a.split)
        run_path = RUNS / ("dense_mxbai_%s_%s.trec" % (dom, a.split))
        if not (ex_path.exists() and (ddir / "documents.jsonl").exists() and run_path.exists()):
            print("[skip] %s" % dom, flush=True)
            continue
        corpus = {}
        with open(ddir / "documents.jsonl", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                d = json.loads(ln)
                corpus[d["id"]] = (d.get("title", "") + ". " + d.get("content", "")).strip()
        examples, order = {}, []
        with open(ex_path, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                examples[r["id"]] = r["query"]
                order.append(r["id"])
        if qids_filter:
            order = [q for q in order if q in qids_filter]
        qrels = read_qrels(ddir / ("qrels_%s.txt" % a.split))
        runs = read_trec(run_path)
        sel = [q for q in order if q in runs][: a.per_domain]
        for qid in sel:
            query = query_text(examples[qid])
            docs = [d for d, _ in runs[qid][: a.pool]]
            scores, gens = {}, []
            for start in range(0, len(docs), a.group):
                chunk = docs[start:start + a.group]
                passages = fmt_passages(chunk, corpus, a.doc_words)
                text = USER_PROMPT.format(TOPK=len(chunk), QUERY=query, PASSAGES=passages)
                msgs = [{"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": text}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                if a.prefill:
                    prompt = prompt + "<answer>" + CH_NL + "```json" + CH_NL + "{"
                gen = model_scores(model, tok, prompt, a.max_new_tokens)
                n_prompts += 1
                parsed = parse_scores(gen)
                vals = [parsed.get(i + 1, -1.0) for i in range(len(chunk))]
                if a.zscore_groups and len(vals) > 1:
                    ok = [v for v in vals if v >= 0]
                    mu = sum(ok) / len(ok) if ok else 0.0
                    sd = ((sum((v - mu) ** 2 for v in ok) / len(ok)) ** 0.5 if len(ok) > 1 else 1.0) or 1.0
                    vals = [((v - mu) / sd if v >= 0 else -3.0) for v in vals]
                for i, d in enumerate(chunk):
                    scores[d] = vals[i]
                gens.append({"group_start": start, "n": len(chunk), "parsed": len(parsed), "gen": gen[:300]})
            if a.cascade:
                survivors = []
                for start in range(0, len(docs), a.group):
                    chunk = docs[start:start + a.group]
                    survivors.extend(sorted(chunk, key=lambda d: -scores.get(d, -1.0))[: a.cascade_keep])
                final = survivors[: a.group * 2]
                text = USER_PROMPT.format(TOPK=len(final), QUERY=query,
                                          PASSAGES=fmt_passages(final, corpus, a.doc_words))
                msgs = [{"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": text}]
                prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                if a.prefill:
                    prompt = prompt + "<answer>" + CH_NL + "```json" + CH_NL + "{"
                gen = model_scores(model, tok, prompt, a.max_new_tokens)
                n_prompts += 1
                parsed = parse_scores(gen)
                for i, d in enumerate(final):
                    scores[d] = float(parsed.get(i + 1, -1.0))
                gens.append({"cascade": True, "n": len(final), "parsed": len(parsed), "gen": gen[:300]})
            ranked = sorted(docs, key=lambda d: (-scores.get(d, -1.0), docs.index(d)))
            gold = qrels.get(qid, set())
            nd = ndcg_at_k(ranked, gold, 10)
            base = ndcg_at_k(docs, gold, 10)
            per_query.append({"domain": dom, "qid": qid, "n_docs": len(docs), "query": query[:400],
                              "dense_ndcg10": base, "diver_ndcg10": nd,
                              "n_relevant_top10_dense": sum(1 for d in docs[:10] if d in gold),
                              "n_relevant_top10_diver": sum(1 for d in ranked[:10] if d in gold),
                              "ranked_top10": ranked[:10], "dense_top10": docs[:10],
                              "groups": gens})
            print("%s %s dense %.4f diver %.4f" % (dom, qid, base, nd), flush=True)
            if a.limit_queries and len(per_query) >= a.limit_queries:
                break

    by = defaultdict(list)
    for r in per_query:
        by[r["domain"]].append(r)

    def macro(key):
        if not by:
            return 0.0
        return sum(sum(x[key] for x in v) / len(v) for v in by.values()) / len(by)

    payload = {"model": a.model, "split": a.split, "group": a.group, "doc_words": a.doc_words,
               "decoding": "greedy, answer prefilled" if a.prefill else "greedy (card uses temperature 0.3 sampling)",
               "pool": a.pool, "zscore_groups": bool(a.zscore_groups), "cascade": bool(a.cascade), "prompts": n_prompts,
               "seconds": round(time.time() - t0, 1), "n_queries": len(per_query),
               "dense_macro": round(macro("dense_ndcg10"), 6), "diver_macro": round(macro("diver_ndcg10"), 6),
               "zero_dense": sum(1 for r in per_query if r["dense_ndcg10"] == 0),
               "zero_diver": sum(1 for r in per_query if r["diver_ndcg10"] == 0),
               "rel_top10_dense": sum(r["n_relevant_top10_dense"] for r in per_query),
               "rel_top10_diver": sum(r["n_relevant_top10_diver"] for r in per_query),
               "per_query": per_query}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("RESULT dense %.6f diver %.6f zeros %d->%d rel@10 %d->%d" % (
        payload["dense_macro"], payload["diver_macro"], payload["zero_dense"], payload["zero_diver"],
        payload["rel_top10_dense"], payload["rel_top10_diver"]))


if __name__ == "__main__":
    main()
