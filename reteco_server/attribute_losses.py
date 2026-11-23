"""Attribute the six-pair losses (reviewer item iii).

For every dev query that has a beneficial action inside the six head-first pairs,
classify what the frozen rule did and emit the text the judge actually saw:

  positive      the swap it made helped
  missed_swap   no acceptable beneficial action at all (rule never fires)
  wrong_choice  a beneficial action was acceptable but a different one was picked

Each record carries the query, and for every one of the six pairs the slot, the
true delta, the merged probability, the two-order agreement, and the 112-word
window that the judge saw for both documents.  Goal: separate "the evidence was
never in the window", "the document answers only one facet", "the judgement is
wrong on a document that clearly answers", and "the label is questionable".

Run with RETECO_DATA_ROOT pointing at the corpus (documents.jsonl are not in the
repo).  Output: results/replacement_v2/loss_attribution_dev.json
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("RETECO_DATA_ROOT", r"D:/RETECO/data"))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_pairwise_replacement_v2 as B  # noqa: E402

WIN, STRIDE = 112, 80


def load_domain(domain, wanted):
    docs = {}
    path = DATA / "track1_tempo" / domain / "documents.jsonl"
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            did = str(d.get("id", d.get("docid", d.get("_id"))))
            if did in wanted:
                docs[did] = (d.get("title", "") + ". " + (d.get("content") or d.get("text") or "")).strip()
                if len(docs) == len(wanted):
                    break
    return docs


def load_queries(domain):
    out = {}
    with open(DATA / "track1_tempo" / domain / "examples_dev.jsonl", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            out[str(r.get("id"))] = (r.get("query") or r.get("text") or "")
    return out


def main():
    d = json.loads((ROOT / "results/replacement_v2/expanded3_fixed_dev.json").read_text(encoding="utf-8"))
    rows = d["budgets"]["6"]["per_query"]

    need = defaultdict(set)
    plans = []
    for r in rows:
        acts = sorted(r["actions"], key=lambda a: (a["slot"], -a["prior"]))[:6]
        if not any(a["delta"] > 1e-12 for a in acts):
            continue
        elig = [a for a in acts if a["agree"] and a["mp"] - 0.5 > 0]
        pick = max(elig, key=lambda a: a["mp"]) if elig else None
        dv = pick["delta"] if pick else 0.0
        cls = "positive" if dv > 1e-12 else ("missed_swap" if not elig else "wrong_choice")
        plans.append({"domain": r["domain"], "qid": r["qid"], "class": cls,
                      "baseline": r["baseline"], "realised_delta": dv,
                      "best_available": max(a["delta"] for a in acts), "actions": acts})
        for a in acts:
            need[r["domain"]].add(str(a["incumbent"]))
            need[r["domain"]].add(str(a["challenger"]))

    texts, queries = {}, {}
    for dom, wanted in need.items():
        texts[dom] = load_domain(dom, wanted)
        queries[dom] = load_queries(dom)
        print(dom, "docs", len(texts[dom]), "wanted", len(wanted), flush=True)

    out = []
    for p in plans:
        dom = p["domain"]
        q = queries[dom].get(str(p["qid"]), "")
        qterms = B.terms(q)
        rec = {k: p[k] for k in ("domain", "qid", "class", "baseline", "realised_delta", "best_available")}
        rec["query"] = q
        rec["pairs"] = []
        for a in p["actions"]:
            inc = texts[dom].get(str(a["incumbent"]), "")
            ch = texts[dom].get(str(a["challenger"]), "")
            rec["pairs"].append({
                "slot": a["slot"], "delta": a["delta"], "mp": a["mp"], "agree": bool(a["agree"]),
                "acceptable": bool(a["agree"] and a["mp"] - 0.5 > 0),
                "incumbent": a["incumbent"], "challenger": a["challenger"],
                "inc_window": B.passages(inc, qterms, win=WIN, stride=STRIDE)[0] if inc else "",
                "ch_window": B.passages(ch, qterms, win=WIN, stride=STRIDE)[0] if ch else "",
            })
        out.append(rec)

    counts = defaultdict(int)
    best_sum = defaultdict(float)
    for r in out:
        counts[r["class"]] += 1
        best_sum[r["class"]] += max(0.0, r["best_available"])
    payload = {"judge_window": [WIN, STRIDE], "n_queries_with_beneficial": len(out),
               "class_counts": dict(counts),
               "best_available_sum": {k: round(v, 4) for k, v in best_sum.items()},
               "records": out}
    dest = ROOT / "results/replacement_v2/loss_attribution_dev.json"
    dest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("counts", dict(counts))
    print("best-available delta by class", payload["best_available_sum"])
    print("RESULT: attribution_written", dest.stat().st_size)


if __name__ == "__main__":
    main()
