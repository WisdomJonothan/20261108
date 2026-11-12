#!/usr/bin/env python3
"""Aggregate the blinded mechanism codes and compute inter-annotator agreement.

Inputs (results/mismatch/):
  annotator_sheet.jsonl  -- 53 blinded items, sheet order
  partA/B/C.jsonl        -- primary coding, disjoint ranges
  partD_second.jsonl     -- independent second pass over the first 20 items
  key.json               -- provenance (used only for the domain breakdown)

    python scripts/aggregate_mismatch.py --out results/mismatch/summary.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(HERE))


def read_jsonl(p):
    return [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]


def kappa(pairs):
    """Cohen's kappa for two label lists."""
    n = len(pairs)
    if n == 0:
        return None
    obs = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    exp = sum(ca[k] / n * cb[k] / n for k in set(ca) | set(cb))
    return None if exp == 1 else (obs - exp) / (1 - exp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "mismatch"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "mismatch", "summary.json"))
    args = ap.parse_args()
    d = args.dir
    sheet = read_jsonl(os.path.join(d, "annotator_sheet.jsonl"))
    primary = {}
    for name in ("partA.jsonl", "partB.jsonl", "partC.jsonl"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            for r in read_jsonl(p):
                primary[r["item_id"]] = dict(r, part=name)
    second = {}
    p2 = os.path.join(d, "partD_second.jsonl")
    if os.path.isfile(p2):
        for r in read_jsonl(p2):
            second[r["item_id"]] = r
    key = json.load(open(os.path.join(d, "key.json"), encoding="utf-8"))

    missing = [it["item_id"] for it in sheet if it["item_id"] not in primary]
    dist = Counter(r["mechanism"] for r in primary.values())
    by_domain = defaultdict(Counter)
    for iid, r in primary.items():
        by_domain[key[iid]["domain"]][r["mechanism"]] += 1
    conf = Counter(r["confidence"] for r in primary.values())

    flt = os.path.join(d, "def2_filter.json")
    surv = set()
    if os.path.isfile(flt):
        surv = set(json.load(open(flt, encoding="utf-8"))["survivors"])
    dist_def2 = Counter(r["mechanism"] for i, r in primary.items() if i in surv)
    pairs = [(primary[i]["mechanism"], second[i]["mechanism"])
             for i in second if i in primary]
    agree = sum(1 for a, b in pairs if a == b)
    k = kappa(pairs)
    disagree = [{"item_id": i, "pass1": primary[i]["mechanism"],
                 "pass2": second[i]["mechanism"],
                 "domain": key[i]["domain"]} for i in second
                if i in primary and primary[i]["mechanism"] != second[i]["mechanism"]]

    out = {
        "n_items": len(sheet),
        "n_coded": len(primary),
        "missing": missing,
        "mechanism_distribution": dict(dist.most_common()),
        "mechanism_share": {k2: round(v / max(1, len(primary)), 4)
                            for k2, v in dist.most_common()},
        "confidence": dict(sorted(conf.items())),
        "by_domain": {k2: dict(v) for k2, v in by_domain.items()},
        "def2_strict": {
            "n": len(surv),
            "mechanism_distribution": dict(dist_def2.most_common()),
            "mechanism_share": {k2: round(v / max(1, len(surv)), 4)
                                for k2, v in dist_def2.most_common()},
        },
        "agreement": {
            "n": len(pairs),
            "observed": round(agree / len(pairs), 4) if pairs else None,
            "cohen_kappa": round(k, 4) if k is not None else None,
            "disagreements": disagree,
        },
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k2: out[k2] for k2 in
                      ("n_items", "n_coded", "mechanism_distribution",
                       "mechanism_share", "agreement")}, indent=1, ensure_ascii=False))
    if missing:
        print("MISSING:", missing)


if __name__ == "__main__":
    sys.exit(main())
