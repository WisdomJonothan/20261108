"""Step-level support judgment, and the stability test the review asked for.

For a sampled set of decomposition steps we ask one 4B model, in a single
prompt, whether a passage satisfies *this specific evidence requirement*
(S = supports, C = contradicts, I = insufficient) and compare it with the same
model's global "does this passage help answer the question" judgment.

Three quantities come out:
  * discrimination  - P(S | gold document) vs P(S | dense top-10 non-gold)
  * localisation    - the same comparison for the global question prompt
  * stability       - for queries with two or more steps, a document that is the
                      gold of step A is judged again under step B.  A prompt that
                      really conditions on the requirement should flip to I/C for
                      most such cross-assignments while keeping S for step A.

Everything is measured on train queries (dev stays confirmation-only), and the
negative documents are unjudged-by-construction only if the reviewer's concern is
accepted; here they are dense top-10 documents that are not in any gold list of
the query, which is stated in the payload.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ir_metrics import ndcg_at_k  # noqa: F401,E402  (import kept for parity of env)

DATA = Path("/homeC/wujiaheng/RETECO/data/track1_tempo")
RUNS = Path("/homeC/wujiaheng/RETECO/runs")
CH = chr(10)

SYS = ("You are grading whether one passage satisfies one specific evidence "
       "requirement of a search question. Judge only the requirement given, not "
       "the question as a whole.")

STEP_PROMPT = CH.join([
    "Question: {q}",
    "",
    "Evidence requirement: {step}",
    "",
    "Passage:",
    "{doc}",
    "",
    "Does the passage contain the specific evidence this requirement asks for?",
    "S = yes, it contains that evidence",
    "I = no, it does not",
    "Answer with one letter.",
])

GLOBAL_PROMPT = CH.join([
    "Question: {q}",
    "",
    "Passage:",
    "{doc}",
    "",
    "Does the passage contain evidence that helps answer the question?",
    "Y = yes",
    "N = no",
    "Answer with one letter.",
])


def clean(t):
    import re
    t = re.sub(r"<[^>]+>", " ", t or "")
    return re.sub(r"\s+", " ", t).strip()


def read_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                out.append(json.loads(ln))
    return out


def letter_probs(model, tok, prompt, letters):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    with torch.inference_mode():
        out = model(**ids).logits[0, -1].float()
    probs = torch.softmax(out, dim=-1)
    got = {}
    for L in letters:
        tid = tok.encode(L)[0]
        got[L] = float(probs[tid])
    s = sum(got.values()) or 1.0
    return {k: v / s for k, v in got.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/reteco/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--split", default="train")
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--docs-words", type=int, default=220)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out", default="results/analysis/step_support.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).eval().to("cuda")

    queries, pool = [], []
    for ddir in sorted(DATA.iterdir()):
        steps_f = ddir / ("steps_%s.jsonl" % a.split)
        docs_f = ddir / "documents.jsonl"
        run_f = RUNS / ("dense_mxbai_%s_%s.trec" % (ddir.name, a.split))
        if not (steps_f.exists() and docs_f.exists() and run_f.exists()):
            continue
        steps = {r["id"]: r for r in read_jsonl(steps_f)}
        runs = defaultdict(list)
        with open(run_f, encoding="utf-8") as f:
            for ln in f:
                p = ln.split()
                if len(p) >= 5:
                    runs[p[0]].append((p[2], float(p[4])))
        for q in runs:
            runs[q].sort(key=lambda x: -x[1])
        keep = defaultdict(set)
        for q, rec in steps.items():
            for s in rec["steps"]:
                keep[q].update(s.get("gold_ids") or [])
        for q, rec in steps.items():
            if q not in runs or len(rec["steps"]) < 2:
                continue
            gold_all = keep[q]
            negs = [d for d, _ in runs[q][20:100] if d not in gold_all]
            if not negs:
                continue
            pool.append({"domain": ddir.name, "qid": q, "rec": rec, "negs": negs[:2]})
    rng.shuffle(pool)
    pool = pool[: a.queries]

    texts, records = {}, []
    for item in pool:
        ddir = DATA / item["domain"]
        need = set()
        for s in item["rec"]["steps"]:
            need.update([str(x) for x in (s.get("gold_ids") or [])])
        need.update(item["negs"])
        if item["domain"] not in texts:
            texts[item["domain"]] = {}
        missing = need - set(texts[item["domain"]])
        if missing:
            with open(ddir / "documents.jsonl", encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    d = json.loads(ln)
                    did = str(d.get("id"))
                    if did in missing:
                        texts[item["domain"]][did] = clean(d.get("title", "") + ". " + d.get("content", ""))[: a.docs_words * 8]
                        missing.discard(did)
                        if not missing:
                            break
        q = clean(item["rec"]["query"])
        steps = [s for s in item["rec"]["steps"] if s.get("gold_ids")]
        if len(steps) < 2:
            continue
        for s in steps:
            pos = str(s["gold_ids"][0])
            ptext = texts[item["domain"]].get(pos, "")
            if not ptext:
                continue
            rec = {"domain": item["domain"], "qid": item["qid"], "step": s["step"],
                   "step_id": s.get("step_id"), "requirement": s["step_instruction"],
                   "pos_doc": pos, "judgments": {}}
            j = letter_probs(model, tok, STEP_PROMPT.format(q=q, step=s["step_instruction"], doc=ptext), "SI")
            g = letter_probs(model, tok, GLOBAL_PROMPT.format(q=q, doc=ptext), "YN")
            rec["judgments"]["pos_under_own_step"] = j
            rec["judgments"]["pos_under_global"] = g
            for other in steps:
                if other is s:
                    continue
                jo = letter_probs(model, tok, STEP_PROMPT.format(q=q, step=other["step_instruction"], doc=ptext), "SI")
                rec["judgments"]["pos_under_other_step"] = {"step": other["step"], "probs": jo}
                break
            if item["negs"]:
                ndoc = item["negs"][0]
                ntext = texts[item["domain"]].get(ndoc, "")
                if ntext:
                    rec["neg_doc"] = ndoc
                    rec["judgments"]["neg_under_own_step"] = letter_probs(
                        model, tok, STEP_PROMPT.format(q=q, step=s["step_instruction"], doc=ntext), "SI")
                    rec["judgments"]["neg_under_global"] = letter_probs(
                        model, tok, GLOBAL_PROMPT.format(q=q, doc=ntext), "YN")
            records.append(rec)
            print("%s %s step %s S=%.2f (own) S=%.2f (other) | neg S=%.2f" % (
                rec["domain"], rec["qid"], rec["step"],
                rec["judgments"]["pos_under_own_step"]["S"],
                rec["judgments"].get("pos_under_other_step", {}).get("probs", {}).get("S", float("nan")),
                rec["judgments"].get("neg_under_own_step", {}).get("S", float("nan"))), flush=True)

    def mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "n_step_records": len(records),
        "pos_S_own": round(mean(r["judgments"]["pos_under_own_step"]["S"] for r in records), 4),
        "neg_S_own": round(mean(r["judgments"]["neg_under_own_step"]["S"] for r in records if "neg_under_own_step" in r["judgments"]), 4),
        "pos_Y_global": round(mean(r["judgments"]["pos_under_global"]["Y"] for r in records), 4),
        "neg_Y_global": round(mean(r["judgments"]["neg_under_global"]["Y"] for r in records if "neg_under_global" in r["judgments"]), 4),
        "pos_S_other_step": round(mean(r["judgments"]["pos_under_other_step"]["probs"]["S"] for r in records if "pos_under_other_step" in r["judgments"]), 4),
        "flip_rate": round(mean(1.0 if (r["judgments"].get("pos_under_other_step", {}).get("probs", {}).get("S", 1.0) < 0.5) else 0.0 for r in records), 4),
        "own_argmax_S": round(mean(1.0 if r["judgments"]["pos_under_own_step"]["S"] > 0.5 else 0.0 for r in records), 4),
    }
    payload = {"model": a.model, "split": a.split, "seed": a.seed, "docs_words": a.docs_words,
               "note": ("negative documents are dense rank 21-100 documents that appear in no gold list "
                        "of the query; step judgments use a single binary prompt with the requirement in it"),
               "summary": summary, "records": records}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("RESULT " + json.dumps(summary))


if __name__ == "__main__":
    main()
