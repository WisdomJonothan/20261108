#!/usr/bin/env python3
"""Attribute the fold losses: was the decisive evidence cut away, or judged wrong?

For every query in the three held-out folds this script reconstructs the six
judged pairs, replays the frozen acceptance rule, and classifies the outcome:

  positive      the rule swapped and the swap helped
  missed_swap   a beneficial pair existed but the rule accepted none
  wrong_choice  the rule swapped a pair that did not help while a better one existed

For each affected query it measures whether the requirement-bearing text of the
two documents was actually inside the 112-word window the judge was shown:

  cov_shown     condition units found in the shown window
  cov_best      condition units in the best 112-word window of that document
  outside       cov_best - cov_shown  (units of decisive text the judge never saw)

Condition units are numbers/dates, capitalised entities and requirement content
terms, i.e. the parts of a step instruction that can be checked against a
passage.  The output is a worksheet for human adjudication plus a distribution,
not a substitute for it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")


def load_modules(root: Path):
    sys.path.insert(0, str(root / "scripts"))
    import benchmark_pairwise_replacement_v2 as B
    return B


NUM = re.compile(r"\b\d[\d.,]*\b")
CAP = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


def units(text):
    """Condition units: digits and capitalised tokens (cheap, language-agnostic)."""
    return set(NUM.findall(text)) | set(CAP.findall(text))


def read_steps(directory: Path, split: str) -> dict:
    out = {}
    path = directory / f"steps_{split}.jsonl"
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        item = json.loads(ln)
        steps = [s.get("step_instruction") or "" for s in item.get("steps", [])]
        steps = [s.strip() for s in steps if s and s.strip()]
        if steps:
            out[str(item["id"])] = steps
    return out


def windows(body: str, win: int = 112, stride: int = 80):
    words = body.split()
    return [" ".join(words[i:i + win]) for i in range(0, len(words), stride)] or [""]


def coverage(window: str, req_units: set) -> int:
    return len(units(window) & req_units)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--folds", default="f0,f1,f2")
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--out", default="results/analysis/req_attribution_folds.json")
    ap.add_argument("--worksheet", default="results/analysis/req_attribution_sheet.csv")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    B = load_modules(root)
    rows = []
    for fold in a.folds.split(","):
        d = json.load(open(root / "results/replacement_v2" /
                           f"expanded3_fixed_train_{fold}.json", encoding="utf-8"))
        for r in d["budgets"]["18"]["per_query"]:
            r["fold"] = fold
            rows.append(r)
    print("queries", len(rows))

    # replay the six-pair rule
    for r in rows:
        pool = sorted(r["actions"], key=lambda x: (x["slot"], -x["prior"]))[:6]
        elig = [x for x in pool if x.get("agree") and x["mp"] > 0.5]
        best = max(elig, key=lambda x: x["mp"]) if elig else None
        r["six"] = pool
        r["chosen"] = best
        r["best_beneficial"] = max(pool, key=lambda x: x["delta"])
        r["outcome"] = ("positive" if best and best["delta"] > 1e-9 else
                        "harmful" if best and best["delta"] < -1e-9 else
                        "missed_swap" if r["best_beneficial"]["delta"] > 1e-9 else
                        "neutral")
    tally = Counter(r["outcome"] for r in rows)
    print("outcome tally", dict(tally))

    # documents and requirement text
    corpus = {}
    steps = {}
    query_text = defaultdict(dict)
    for dom in {r["domain"] for r in rows}:
        directory = root / "data/track1_tempo" / dom
        steps[dom] = read_steps(directory, "train")
        for ln in (directory / "examples_train.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                ex = json.loads(ln)
                query_text[dom][str(ex["id"])] = ex["query"]
        need = set()
        for r in rows:
            if r["domain"] != dom:
                continue
            for act in r["six"]:
                need.add(act["incumbent"])
                need.add(act["challenger"])
        got = {}
        with (directory / "documents.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line)
                if item["id"] in need:
                    got[item["id"]] = item["content"]
                    if len(got) == len(need):
                        break
        corpus.update({(dom, k): v for k, v in got.items()})
        print(f"  {dom}: {len(got)}/{len(need)} docs")

    cases = []
    for r in rows:
        dom = r["domain"]
        st = steps[dom].get(str(r["qid"]), [])
        if not st:
            continue
        req = st[0]
        req_units = units(req)
        if r["outcome"] == "positive":
            act = r["chosen"]
        elif r["outcome"] in ("missed_swap", "harmful"):
            act = r["best_beneficial"]
        else:
            continue
        inc, ch = act["incumbent"], act["challenger"]
        if (dom, inc) not in corpus or (dom, ch) not in corpus:
            continue
        rec = {"fold": r["fold"], "domain": dom, "qid": r["qid"], "outcome": r["outcome"],
               "slot": act["slot"], "incumbent": inc, "challenger": ch,
               "delta_ndcg": act["delta"], "mp": act["mp"], "agree": bool(act["agree"]),
               "requirement": req}
        for tag, docid in (("inc", inc), ("ch", ch)):
            body = corpus[(dom, docid)]
            qt = B.terms(query_text[dom].get(str(r["qid"]), req))
            req_terms = B.terms(req) - qt
            shown = B.passages(body, qt)[0]
            ws = windows(body)
            covs = [coverage(w, req_units) for w in ws]
            rec[f"{tag}_cov_shown"] = coverage(shown, req_units)
            rec[f"{tag}_cov_best"] = max(covs) if covs else 0
            rec[f"{tag}_outside"] = rec[f"{tag}_cov_best"] - rec[f"{tag}_cov_shown"]
            term_shown = len(B.terms(shown) & req_terms)
            term_best = max(len(B.terms(w) & req_terms) for w in ws) if ws else 0
            rec[f"{tag}_req_terms_shown"] = term_shown
            rec[f"{tag}_req_terms_best"] = term_best
            rec[f"{tag}_req_elsewhere"] = bool(req_terms and term_best > term_shown)
            rec[f"{tag}_text"] = shown[:600]
        rec["evidence_outside"] = (rec["ch_outside"] > rec["inc_outside"] + 1)
        rec["req_text_elsewhere"] = bool(rec["ch_req_elsewhere"] or rec["inc_req_elsewhere"])
        cases.append(rec)

    by = defaultdict(list)
    for c in cases:
        by[c["outcome"]].append(c)
    summary = {}
    for outcome, items in by.items():
        n = len(items)
        outside = sum(1 for c in items if c["evidence_outside"])
        elsewhere = sum(1 for c in items if c["req_text_elsewhere"])
        summary[outcome] = {
            "n": n,
            "challenger_evidence_outside_window": outside,
            "share": round(outside / n, 4) if n else 0.0,
            "requirement_text_elsewhere": elsewhere,
            "share_requirement_elsewhere": round(elsewhere / n, 4) if n else 0.0,
            "mean_ch_outside": round(sum(c["ch_outside"] for c in items) / n, 3),
            "mean_inc_outside": round(sum(c["inc_outside"] for c in items) / n, 3),
            "mean_delta": round(sum(c["delta_ndcg"] for c in items) / n, 4),
        }
    print(json.dumps(summary, indent=1))

    out = {"tally": dict(tally), "case_summary": summary,
           "note": ("Condition units are digits and capitalised tokens from the first step "
                    "instruction. evidence_outside=True means the challenger's requirement-"
                    "bearing text lies further outside the shown window than the incumbent's.")}
    (root / a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(root / a.out, "w"), ensure_ascii=False, indent=1)

    with open(root / a.worksheet, "w", encoding="utf-8") as fh:
        fh.write("fold,domain,qid,outcome,slot,delta,mp,agree,req_elsewhere_ch,"
                 "req_elsewhere_inc,challenger_better,human_verdict,requirement,"
                 "challenger_window,incumbent_window\n")
        for o, items in by.items():
            for c in items[:a.per_class]:
                def clean(s):
                    return (s or "").replace(",", ";").replace("\n", " ")[:400]
                fh.write(f"{c['fold']},{c['domain']},{c['qid']},{o},{c['slot']},"
                         f"{c['delta_ndcg']:.4f},{c['mp']:.3f},{c['agree']},"
                         f"{c['ch_req_elsewhere']},{c['inc_req_elsewhere']},,,"
                         f"{clean(c['requirement'])},{clean(c['ch_text'])},"
                         f"{clean(c['inc_text'])}\n")
    print("wrote", a.out, a.worksheet)


if __name__ == "__main__":
    main()
