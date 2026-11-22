#!/usr/bin/env python3
"""Expanded action space with a CE-augmented candidate pool.

Candidates = union(dense top-100, frozen full-query CE top-100) minus the frozen
top-10, ordered by best rank in either source (dense before CE on ties), keep 20.
Actions = (slot 1..10) x candidates, ordered by the cheap gap
    prior = (dense_score_cand - dense_score_inc) + (ce_z_cand - ce_z_inc),
divided by log2(slot + 2); each action is judged in both orders with the Direct
prompt; budgets are replayed from the same judgement cache.

`--order random` gives the matched-budget control.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import benchmark_pairwise_replacement_v2 as B

DOMAINS = B.DOMAINS


def read_run(path):
    out = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        f = ln.split()
        out.setdefault(f[0], {})[f[2]] = (int(f[3]), float(f[4]))
    return out


def zmap(scores):
    v = np.array(list(scores.values()))
    mu, sd = float(v.mean()), float(v.std() or 1.0)
    return {d: (s - mu) / sd for d, s in scores.items()}


def build(root, max_actions, order, rng, ce_tag, start_tag="sub_sft_e2bc", split="dev", qids=None):
    rows = []
    for domain in DOMAINS:
        directory = Path(root) / "data/track1_tempo" / domain
        dense = read_run(Path(root) / f"runs/dense_mxbai_{domain}_{split}.trec")
        frozen = read_run(Path(root) / f"runs/{start_tag}_{domain}_{split}.trec")
        ce = read_run(Path(root) / f"runs/{ce_tag}_{domain}_{split}_ce.trec")
        need = set()
        for qid, docs in frozen.items():
            need |= {d for d, (r, _s) in docs.items() if r <= 10}
            need |= {d for d, (r, _s) in dense.get(qid, {}).items() if r <= 100}
            need |= {d for d, (r, _s) in ce.get(qid, {}).items() if r <= 100}
        corpus = {}
        with (directory / "documents.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                it = json.loads(line)
                if it["id"] in need:
                    corpus[it["id"]] = it["content"]
                    if len(corpus) == len(need):
                        break
        gold = B.read_qrels(directory / f"qrels_{split}.txt")
        examples = {}
        for ln in (directory / f"examples_{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                ex = json.loads(ln)
                examples[ex["id"]] = ex["query"]
        for qid in sorted(set(frozen) & set(examples)):
            if qids is not None and str(qid) not in qids:
                continue
            top = [d for d, (r, _s) in sorted(frozen[qid].items(), key=lambda kv: kv[1][0])][:10]
            if len(top) < 10:
                continue
            query = B.strip_html(examples[qid])
            qt = B.terms(query)
            dmap = dense.get(qid, {})
            cmap = ce.get(qid, {})
            best = {}
            for d, (r, _s) in dmap.items():
                if r <= 100 and d not in set(top):
                    best[d] = (r, 0)
            for d, (r, _s) in cmap.items():
                if r <= 100 and d not in set(top):
                    if d not in best or (r, 1) < best[d]:
                        best[d] = (r, 1)
            cands = [d for d, _ in sorted(best.items(), key=lambda kv: kv[1])][:20]
            cez_all = zmap({d: s for d, (_r, s) in cmap.items()})
            cez = {d: cez_all.get(d, -3.0) for d in cmap}
            texts = {d: B.passages(corpus.get(d, ""), qt)[0] for d in set(top) | set(cands)}
            acts = []
            for slot in range(1, 11):
                inc = top[slot - 1]
                di = dmap.get(inc, (999, -9.0))[1]
                ci = cez.get(inc, -3.0)
                for c in cands:
                    gap = (dmap.get(c, (999, -9.0))[1] - di) + (cez.get(c, -3.0) - ci)
                    acts.append({"slot": slot, "incumbent": inc, "challenger": c,
                                 "inc_text": texts[inc], "ch_text": texts[c],
                                 "prior": gap / math.log2(slot + 2)})
            if order == "random":
                rng.shuffle(acts)
            else:
                acts.sort(key=lambda x: -x["prior"])
            g = gold.get(qid, set())
            rows.append({"domain": domain, "qid": qid, "query": query, "top": top, "gold": g,
                         "baseline": B.ndcg10(top, g), "actions": acts[:max_actions]})
    return rows


@torch.inference_mode()
def score(model, tok, rows, batch_size, option_ids):
    prompts, index = [], []
    for ri, row in enumerate(rows):
        for ai, act in enumerate(row["actions"]):
            for rev in (False, True):
                pair = {"incumbent": act["incumbent"], "challenger": act["challenger"],
                        "incumbent_text": act["inc_text"], "challenger_text": act["ch_text"],
                        "retained": []}
                text = B.prompt(row, pair, False, rev)
                text = tok.apply_chat_template([{"role": "user", "content": text}],
                                               tokenize=False, add_generation_prompt=True)
                prompts.append(text)
                index.append((ri, ai, rev))
    out = []
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], padding=True,
                  add_special_tokens=False, return_tensors="pt").to("cuda")
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, option_ids]
        out.extend(logits.float().softmax(-1).cpu().tolist())
    for (ri, ai, rev), p in zip(index, out):
        # in the reversed prompt A is the challenger, so the option order is swapped back
        names = ("incumbent", "challenger", "abstain") if not rev else (
            "challenger", "incumbent", "abstain")
        rows[ri]["actions"][ai].setdefault("p", {})["rev" if rev else "fwd"] = {
            n: float(v) for n, v in zip(names, p)}
    return len(prompts)


def decide(row, budget):
    best = None
    for act in row["actions"][:budget]:
        p = act.get("p")
        if not p:
            continue
        mp = (p["fwd"]["challenger"] + p["rev"]["challenger"]) / 2
        agree = (max(p["fwd"], key=p["fwd"].get) == "challenger"
                 and max(p["rev"], key=p["rev"].get) == "challenger")
        if agree and mp > 0.5 and (best is None or mp > best[0]):
            best = (mp, act)
    return best


def evaluate(rows, budget):
    per, counts = [], {"helpful": 0, "harmful": 0, "neutral": 0, "no_swap": 0}
    for row in rows:
        best = decide(row, budget)
        ranked = list(row["top"])
        if best:
            ranked[best[1]["slot"] - 1] = best[1]["challenger"]
        pol = B.ndcg10(ranked, row["gold"])
        d = pol - row["baseline"]
        if ranked == row["top"]:
            counts["no_swap"] += 1
        elif d > 1e-12:
            counts["helpful"] += 1
        elif d < -1e-12:
            counts["harmful"] += 1
        else:
            counts["neutral"] += 1
        acts = []
        for act in row["actions"]:
            p_ = act.get("p")
            if not p_:
                continue
            mp = (p_["fwd"]["challenger"] + p_["rev"]["challenger"]) / 2
            agree = (max(p_["fwd"], key=p_["fwd"].get) == "challenger"
                     and max(p_["rev"], key=p_["rev"].get) == "challenger")
            alt = list(row["top"])
            alt[act["slot"] - 1] = act["challenger"]
            acts.append({"slot": act["slot"], "incumbent": act["incumbent"],
                         "challenger": act["challenger"], "prior": act["prior"],
                         "mp": mp, "agree": agree,
                         "p_fwd": p_["fwd"]["challenger"], "p_rev": p_["rev"]["challenger"],
                         "delta": B.ndcg10(alt, row["gold"]) - row["baseline"]})
        per.append({"domain": row["domain"], "qid": row["qid"], "baseline": row["baseline"],
                    "policy": pol, "delta": d, "actions": acts})
    b = B.domain_macro(per, key="baseline")
    p = B.domain_macro(per, key="policy")
    return {"macro_baseline": b, "macro_policy": p, "macro_delta": p - b,
            "swap_counts": counts, "bootstrap": B.paired_bootstrap(per), "per_query": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--model", type=Path, default=Path("/tmp/reteco/models/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--ce-tag", default="fusion100_full_ce_s17_20261115")
    ap.add_argument("--start-tag", default="sub_sft_e2bc")
    ap.add_argument("--max-actions", type=int, default=18)
    ap.add_argument("--budgets", default="2,6,12,18")
    ap.add_argument("--order", default="prior", choices=["prior", "random"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--qids-file", default="")
    ap.add_argument("--out", default="results/replacement_v2/expanded3_dev.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    qids = None
    if a.qids_file:
        _q = json.loads(Path(a.qids_file).read_text(encoding="utf-8"))
        qids = {str(x) for x in (_q.get("qids") if isinstance(_q, dict) else _q)}
    rows = build(a.root, a.max_actions, a.order, random.Random(0), a.ce_tag, a.start_tag, split=a.split, qids=qids)
    if a.limit_queries:
        rows = rows[:a.limit_queries]
    tok = AutoTokenizer.from_pretrained(str(a.model), trust_remote_code=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(a.model), torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True).to("cuda").eval()
    ids = {}
    for opt in "ABC":
        enc = tok.encode(opt, add_special_tokens=False)
        assert len(enc) == 1, (opt, enc)
        ids[opt] = enc[0]
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    n = score(model, tok, rows, a.batch_size, [ids["A"], ids["B"], ids["C"]])
    secs = time.time() - t0
    out = {"order": a.order, "ce_tag": a.ce_tag, "start_tag": a.start_tag, "max_actions": a.max_actions,
           "n_queries": len(rows), "prompts": n, "seconds": secs, "prompts_per_s": n / secs,
           "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20, "budgets": {}}
    for b in [int(x) for x in a.budgets.split(",")]:
        out["budgets"][str(b)] = evaluate(rows, b)
        r = out["budgets"][str(b)]
        print(f"budget {b:2d} macro {r['macro_policy']:.6f} delta {r['macro_delta']:+.6f} "
              f"CI {[round(x, 5) for x in r['bootstrap']['ci95']]} {r['swap_counts']}", flush=True)
    (a.root / a.out).parent.mkdir(parents=True, exist_ok=True)
    (a.root / a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", a.out, n, "prompts", round(secs), "s")


if __name__ == "__main__":
    main()
