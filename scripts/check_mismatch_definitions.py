#!/usr/bin/env python3
"""Re-check the year-mismatch frame under two definitions of consistency.

The interval heuristic in `hardneg_mine.query_interval` keeps only the *largest*
year when the query mentions several (unless an explicit "2015-2017" range
appears).  For a query that says "1945, 1946, ..., 1975" a document about 1946 is
then counted as mismatched although the query names 1946.  This script measures
how many gold documents change side when consistency is instead "the document
year is within +-1 of *any* year the query mentions".

    python scripts/check_mismatch_definitions.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_examples, load_qrels  # noqa: E402
from hardneg_mine import doc_time, load_guidance, query_interval, gap, years_in  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]


def full_years(query, guid):
    txt = [query]
    for k in ("query_summary", "query_temporal_events", "key_time_anchors"):
        v = (guid or {}).get(k)
        if isinstance(v, str):
            txt.append(v)
        elif isinstance(v, list):
            txt.extend(str(x) for x in v if isinstance(x, (str, int)))
    return years_in("\n".join(txt))


def consistent_any(doc_years, qyears):
    """Consistent if any document year falls in the query's anchor span (+-1).

    Using the span (not single anchors) keeps this monotone with the interval
    definition: a document about 2000 is consistent with a "1990-2010" query
    even though neither 1990 nor 2010 is within one year of 2000.
    """
    return gap(doc_years, (min(qyears), max(qyears))) == 0


def main():
    per = {}
    tot = {"gold": 0, "with_year": 0, "mismatch_def1": 0, "mismatch_def2": 0,
           "changed": 0}
    for dom in DOMAINS:
        ddir = os.path.join(ROOT, "data", "track1_tempo", dom)
        ids, docs = load_corpus(ddir)
        docs_by_id = dict(zip(ids, docs))
        qids, queries = load_examples(ddir, "dev")
        gold = load_qrels(ddir, "dev")
        guid = load_guidance(ddir, "dev")
        row = {"gold": 0, "with_year": 0, "mismatch_def1": 0, "mismatch_def2": 0,
               "changed": 0}
        for i, qid in enumerate(qids):
            iv = query_interval(queries[i], guid.get(qid, {}))
            qy = full_years(queries[i], guid.get(qid, {}))
            if not qy:
                continue
            for d in gold.get(qid, set()):
                row["gold"] += 1
                dt = doc_time(docs_by_id.get(d, ""))
                if dt is None:
                    continue
                row["with_year"] += 1
                m1 = iv is not None and gap(dt["years"], iv) > 0
                m2 = not consistent_any(dt["years"], qy)
                row["mismatch_def1"] += int(m1)
                row["mismatch_def2"] += int(m2)
                row["changed"] += int(m1 and not m2)
        per[dom] = row
        for k in row:
            tot[k] += row[k]
    tot["share_def1"] = round(tot["mismatch_def1"] / tot["with_year"], 4)
    tot["share_def2"] = round(tot["mismatch_def2"] / tot["with_year"], 4)
    out = {"per_domain": per, "total": tot}
    json.dump(out, open(os.path.join(ROOT, "results", "mismatch_definitions.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(tot, indent=1))

    # how many of the annotated items survive the stricter definition?
    key = json.load(open(os.path.join(ROOT, "results", "mismatch", "key.json"),
                         encoding="utf-8"))
    survivors, dropped = [], []
    for iid, k in key.items():
        dom, qid, did = k["domain"], k["qid"], k["docid"]
        ddir = os.path.join(ROOT, "data", "track1_tempo", dom)
        ids, docs = load_corpus(ddir)
        docs_by_id = dict(zip(ids, docs))
        qids, queries = load_examples(ddir, "dev")
        guid = load_guidance(ddir, "dev")
        i = qids.index(qid)
        qy = full_years(queries[i], guid.get(qid, {}))
        dt = doc_time(docs_by_id.get(did, ""))
        ok = dt is not None and not consistent_any(dt["years"], qy)
        (survivors if ok else dropped).append(iid)
    print(f"annotated items: {len(key)} survive def2: {len(survivors)} "
          f"artifacts: {len(dropped)}")
    json.dump({"survivors": sorted(survivors), "artifacts": sorted(dropped)},
              open(os.path.join(ROOT, "results", "mismatch", "def2_filter.json"),
                   "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
