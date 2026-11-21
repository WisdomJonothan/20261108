"""Model-assisted annotation of the 100-row error worksheet.

For each row of error_worksheet_dev.csv (domain, qid, query, slot, current doc,
candidate doc, best_delta) ask the judge model which necessary condition of the
query the current document fails, and whether the candidate meets it. The label
is model-generated and must be read as such (the paper says "model-assisted"),
not as human annotation.

python scripts/annotate_worksheet.py --model ../models/Qwen3-4B-Instruct-2507 \
    --in results/replacement_v2/error_worksheet_dev.csv \
    --out results/replacement_v2/error_worksheet_annotated.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["temporal", "exclusion", "entity_scope", "numeric", "comparison", "other", "none"]
SYS = ("You are auditing a retrieval ranking. Given a question and two documents, "
       "say which necessary condition of the question the CURRENT document fails, "
       "choosing exactly one label from: " + ", ".join(LABELS) + ". "
       "Then say, in one short sentence, whether the CANDIDATE satisfies that condition. "
       'Reply with only this JSON object and nothing else: '
       '{"label": "...", "candidate_ok": true, "why": "one sentence"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--in", dest="inp", default="results/replacement_v2/error_worksheet_dev.csv")
    ap.add_argument("--out", default="results/replacement_v2/error_worksheet_annotated.csv")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.inp, encoding="utf-8")))
    texts = {}
    for d in sorted({r["domain"] for r in rows}):
        path = Path("data/track1_tempo") / d / "documents.jsonl"
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                o = json.loads(ln)
                texts[(d, o["id"])] = o.get("content", "")
    if a.limit:
        rows = rows[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to("cuda").eval()

    prompts = []
    for r in rows:
        user = ("Question: %s\n\nCURRENT document (rank %s):\n%s\n\nCANDIDATE document:\n%s"
                % (r["query"], r["slot"], texts.get((r["domain"], r["current"]), "")[:1200], texts.get((r["domain"], r["candidate"]), "")[:1200]))
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))

    outs = []
    for i in range(0, len(prompts), a.batch_size):
        enc = tok(prompts[i:i + a.batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=2048, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False)
        for j in range(gen.shape[0]):
            text = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(text.strip())
        print("annotated %d/%d" % (min(i + a.batch_size, len(prompts)), len(prompts)), flush=True)

    counts = {}
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) +
                           ["label", "candidate_ok", "why", "raw"])
        w.writeheader()
        for r, t in zip(rows, outs):
            m = re.search(r"\{.*\}", t, re.S)
            label, ok, why = "unparsed", "", t[:160].replace("\n", " ")
            if not m:
                low = t.lower()
                for lab in LABELS:
                    if lab in low:
                        label = lab
                        break
            if m:
                try:
                    d = json.loads(m.group(0))
                    label = d.get("label", "unparsed")
                    ok = d.get("candidate_ok", "")
                    why = d.get("why", "")
                except Exception:
                    pass
            counts[label] = counts.get(label, 0) + 1
            r2 = dict(r); r2.update({"label": label, "candidate_ok": ok, "why": why, "raw": t[:200]})
            w.writerow(r2)
    print("labels", counts)
    Path(a.out).with_suffix(".summary.json").write_text(
        json.dumps({"n": len(rows), "labels": counts, "model": a.model,
                    "note": "model-assisted annotation, not human"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
