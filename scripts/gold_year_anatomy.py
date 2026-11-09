#!/usr/bin/env python3
"""How much temporal evidence do *gold* documents carry?

For every dev query with a recoverable time interval we look at its gold
documents and ask: does the document text expose a year at all, and if so does
it agree with the query interval?  This is the denominator behind the
"mismatch is not irrelevance" claim -- and the reason a strict year filter
cannot be the main lever: most gold documents carry no extractable year.

    python scripts/gold_year_anatomy.py --out results/gold_year_anatomy.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_examples, load_qrels  # noqa: E402
from hardneg_mine import doc_time, load_guidance, query_interval, gap  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="results/gold_year_anatomy.json")
    args = ap.parse_args()

    per, tot = {}, {"gold": 0, "with_year": 0, "consistent": 0, "mismatch": 0,
                    "no_interval": 0}
    for dom in DOMAINS:
        ddir = os.path.join(ROOT, "data", "track1_tempo", dom)
        ids, docs = load_corpus(ddir)
        docs_by_id = dict(zip(ids, docs))
        qids, queries = load_examples(ddir, args.split)
        gold = load_qrels(ddir, args.split)
        guid = load_guidance(ddir, args.split)
        row = {"queries": len(qids), "gold": 0, "with_year": 0,
               "consistent": 0, "mismatch": 0, "no_interval": 0,
               "mismatch_frac_of_years": None}
        for qid in qids:
            iv = query_interval(queries[qids.index(qid)], guid.get(qid, {}))
            if iv is None:
                row["no_interval"] += 1
            for d in gold.get(qid, set()):
                row["gold"] += 1
                dt = doc_time(docs_by_id.get(d, ""))
                if dt is None:
                    continue
                row["with_year"] += 1
                if iv is None:
                    continue
                if gap(dt["years"], iv) == 0:
                    row["consistent"] += 1
                else:
                    row["mismatch"] += 1
        n = row["with_year"]
        row["mismatch_frac_of_years"] = round(row["mismatch"] / n, 4) if n else None
        row["year_frac_of_gold"] = round(n / row["gold"], 4) if row["gold"] else None
        per[dom] = row
        for k in ("gold", "with_year", "consistent", "mismatch", "no_interval"):
            tot[k] += row[k]

    tot["year_frac_of_gold"] = round(tot["with_year"] / tot["gold"], 4) if tot["gold"] else None
    tot["mismatch_frac_of_years"] = round(tot["mismatch"] / tot["with_year"], 4) if tot["with_year"] else None
    out = {"split": args.split, "per_domain": per, "total": tot}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(tot, indent=1))
    for d, r in per.items():
        print(f"{d:10s} gold={r['gold']:5d} year={r['with_year']:5d} "
              f"({(r['year_frac_of_gold'] or 0)*100:5.1f}%) cons={r['consistent']:4d} "
              f"mism={r['mismatch']:4d}")


if __name__ == "__main__":
    main()
