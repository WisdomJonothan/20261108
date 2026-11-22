"""Judge the same nine pairs with a condition-aware evidence window.

The frozen harness shows the judge one 112-word window per document, chosen by
overlap with the question's content words (``passages``). This script keeps the
candidate pool, the pairs, the prompt, the model and the decision rule byte-identical
and changes only how the window is chosen:

  overlap : the frozen rule (baseline, for reference)
  cond    : the 112-word window that covers the question's conditions best --
            content terms, numbers/dates quoted in the question, and capitalised
            entity words that appear in the window (same 112-word budget)
  two     : two 56-word windows, the best condition window plus the best
            non-adjacent one (same 112-word budget, spread over the document)

Usage:
  python localized_judge.py --root . --select cond --out results/.../exp_cond_win_dev.json
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmark_pairwise_replacement_v2 as B

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]

NUM = re.compile(r"\d[\d,.]*")
CAP = re.compile(r"\b[A-Z][A-Za-z\-]{2,}\b")


def q_conditions(text):
    return set(NUM.findall(text)), {w.lower() for w in CAP.findall(text)}


DOC_CAP = 4000  # words considered per document; longer pages are navigational anyway


def windows(body, win, stride):
    words = B.strip_html(body).split()[:DOC_CAP]
    return [" ".join(words[i:i + win]) for i in range(0, len(words), stride)] or [""]


def score_window(w, q_terms, nums, caps):
    wl = w.lower()
    s = 1.0 * len(B.terms(w) & q_terms)
    s += 2.0 * sum(1 for n in nums if n in w)
    s += 1.0 * sum(1 for c in caps if c in wl)
    s += 0.001 * min(len(w.split()), 112)
    return s


def select(body, q_text, q_terms, mode, win=112, stride=80):
    if mode == "overlap":
        return B.passages(body, q_terms)[0]
    nums, caps = q_conditions(q_text)
    if mode == "cond":
        ws = windows(body, win, stride)
        return max(ws, key=lambda w: score_window(w, q_terms, nums, caps))
    if mode == "two":
        half = win // 2
        ws = windows(body, half, max(half // 2, 20))
        order = sorted(range(len(ws)),
                       key=lambda i: -score_window(ws[i], q_terms, nums, caps))
        i = order[0]
        j = next((k for k in order if abs(k - i) >= 2), None)
        parts = [ws[i]] + ([ws[j]] if j is not None else [])
        return " ... ".join(parts)
    raise SystemExit("unknown --select %s" % mode)


def load_corpus(directory, need):
    corpus = {}
    with (directory / "documents.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            if item["id"] in need:
                corpus[item["id"]] = item["content"]
                if len(corpus) == len(need):
                    break
    assert not (need - set(corpus)), (directory, len(need - set(corpus)))
    return corpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--candidates", default="results/replacement_v2/candidates_dev.json")
    ap.add_argument("--select", default="cond")
    ap.add_argument("--model", default="models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    t_build = time.perf_counter()
    rows, proto = B.build_rows(root, Path(args.root) / args.candidates, DOMAINS)
    if args.limit_queries:
        rows = rows[:args.limit_queries]
    print("rows=%d build=%.1fs" % (len(rows), time.perf_counter() - t_build), flush=True)

    t0 = time.perf_counter()
    by_domain = {}
    for row in rows:
        by_domain.setdefault(row["domain"], []).append(row)
    for domain, drows in by_domain.items():
        need = set()
        for row in drows:
            need.update(row["top"])
            need.update(p["challenger"] for p in row["pairs"])
        corpus = load_corpus(root / "data/track1_tempo" / domain, need)
        for row in drows:
            qt = B.terms(row["query"])
            mine = set(row["top"]) | {p["challenger"] for p in row["pairs"]}
            cache = {d: select(corpus[d], row["query"], qt, args.select) for d in mine}
            for pair in row["pairs"]:
                pair["incumbent_text"] = cache[pair["incumbent"]]
                pair["challenger_text"] = cache[pair["challenger"]]
            row["doc_text"] = dict(cache)
    prep_s = time.perf_counter() - t0
    print("prep=%.1fs select=%s" % (prep_s, args.select), flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(root / args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(root / args.model, dtype=torch.bfloat16)
    model = model.to(device).eval()
    option_ids = [tok.encode(letter, add_special_tokens=False)[0] for letter in "ABC"]
    assert all(len(tok.encode(l, add_special_tokens=False)) == 1 for l in "ABC"), option_ids
    print("model loaded on %s" % device, flush=True)

    stats = B.score_arm(model, tok, rows, "direct", args.batch_size, device, option_ids)
    print("scored prompts=%d s=%.1f" % (stats["prompts"], stats["seconds_total"]), flush=True)
    per_query = B.apply_and_score(rows, args.threshold)
    del model
    torch.cuda.empty_cache()

    macro = B.domain_macro(per_query, "policy")
    boot = B.paired_bootstrap(per_query)
    base = B.domain_macro(per_query, "baseline")
    counts = {"helpful": 0, "harmful": 0, "neutral": 0, "no_swap": 0}
    gained = lost = 0.0
    for r in per_query:
        if r["swap"] is None:
            counts["no_swap"] += 1
        elif r["delta"] > 1e-9:
            counts["helpful"] += 1
            gained += r["delta"]
        elif r["delta"] < -1e-9:
            counts["harmful"] += 1
            lost += r["delta"]
        else:
            counts["neutral"] += 1
    per_domain = {}
    for r in per_query:
        per_domain.setdefault(r["domain"], []).append(r)
    per_domain = {d: {"n": len(v),
                      "baseline": sum(x["baseline"] for x in v) / len(v),
                      "policy": sum(x["policy"] for x in v) / len(v)}
                  for d, v in per_domain.items()}
    out = {
        "protocol": {"select": args.select, "model": args.model, "threshold": args.threshold,
                     "window_words": 112, "two_window_words": 56, "candidates": args.candidates},
        "macro_policy": macro, "macro_baseline": base, "macro_delta": macro - base,
        "bootstrap": boot, "swap_counts": counts,
        "points_gained": gained, "points_lost": lost,
        "cost": dict(stats, prep_seconds=prep_s),
        "per_domain": per_domain,
        "per_query": per_query,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("select=%s macro=%.6f delta=%+.6f ci=[%+.5f,%+.5f] swaps=%d helpful=%d harmful=%d prompts=%d s=%.1f"
          % (args.select, macro, macro - base, boot["ci95"][0], boot["ci95"][1],
             len(per_query) - counts["no_swap"], counts["helpful"], counts["harmful"],
             stats["prompts"], stats["seconds_total"]))


if __name__ == "__main__":
    main()
