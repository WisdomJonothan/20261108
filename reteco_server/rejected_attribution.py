"""Materials for attributing the six-pair losses (reviewer item iii).

For every dev query that has a beneficial action inside the six head-first pairs,
write one record with the exact text the judge saw:

  query tokenised the same way the judge sees it (title query, as in the frozen run)
  incumbent 112-word window, challenger 112-word window (passages() from the harness)
  slot, delta, merged probability, agreement on both orders, and whether the frozen
  rule would swap.

Queries are split into three classes by what the frozen rule does:
  positive      the swap it made helped
  missed_swap   no acceptable beneficial action (rule never fires)
  wrong_choice  a beneficial action was acceptable but a different action was picked

Output: results/replacement_v2/rejected_attribution_dev.json (+ a markdown digest).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import benchmark_pairwise_replacement_v2 as B  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("RETECO_DATA_ROOT", r"D:/RETECO/data"))
JUDGE_WIN, JUDGE_STRIDE = 112, 80


def query_text(domain, qid):
    p = DATA / "track1_tempo" / domain / "examples_dev.jsonl"
    with open(p, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if str(r.get("qid", r.get("id"))) == str(qid):
                return (r.get("query") or r.get("text") or "").strip()
    return ""


def doc_text(domain, docid):
    p = DATA / "track1_tempo" / domain / "documents.jsonl"
    with open(p, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if str(r.get("id", r.get("docid", r.get("_id")))) == str(docid):
                return (r.get("title", "") + ". " + (r.get("content") or r.get("text") or "")).strip()
    return ""


def main():
    cache = ROOT / "results/replacement_v2/expanded3_fixed_dev.json"
    d = json.loads(cache.read_text(encoding="utf-8"))
    rows = d["budgets"]["6"]["per_query"]
    out = []
    for r in rows:
        acts = sorted(r["actions"], key=lambda a: (a["slot"], -a["prior"]))[:6]
        if not any(a["delta"] > 1e-12 for a in acts):
            continue
        elig = [a for a in acts if a["agree"] and a["mp"] - 0.5 > 0]
        pick = max(elig, key=lambda a: a["mp"]) if elig else None
        dv = pick["delta"] if pick else 0.0
        cls = "positive" if dv > 1e-12 else ("missed_swap" if not elig else "wrong_choice")
        q = query_text(r["domain"], r["qid"])
        terms = B.terms(q)
        rec = {"domain": r["domain"], "qid": r["qid"], "class": cls, "query": q,
               "baseline_ndcg": r["baseline"], "realised_delta": dv,
               "best_available_delta": max(a["delta"] for a in acts),
               "actions": []}
        for a in acts:
            inc = doc_text(r["domain"], a["incumbent"])
            ch = doc_text(r["domain"], a["challenger"])
            rec["actions"].append({
                "slot": a["slot"], "delta": a["delta"], "mp": a["mp"], "agree": a["agree"],
                "p_fwd": a["p_fwd"], "p_rev": a["p_rev"], "acceptable": bool(a["agree"] and a["mp"] - 0.5 > 0),
                "challenger": a["challenger"], "incumbent": a["incumbent"],
                "incumbent_window": B.passages(inc, terms, win=JUDGE_WIN, stride=JUDGE_STRIDE)[0] if inc else "",
                "challenger_window": B.passages(ch, terms, win=JUDGE_WIN, stride=JUDGE_STRIDE)[0] if ch else "",
            })
        out.append(rec)

    counts = defaultdict(int)
    for r in out:
        counts[r["class"]] += 1
    payload = {"judge_window": [JUDGE_WIN, JUDGE_STRIDE], "n_queries_with_beneficial": len(out),
               "class_counts": dict(counts),
               "best_delta_sum_by_class": {c: round(sum(r["best_available_delta"] for r in out if r["class"] == c), 3) for c in counts},
               "records": out}
    dest = ROOT / "results/replacement_v2/rejected_attribution_dev.json"
    dest.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("classes", dict(counts))
    print("best-delta available by class", payload["best_delta_sum_by_class"])
    print("SKIP" if not dest.exists() else "RESULT: attribution_written")


if __name__ == "__main__":
    main()
