#!/usr/bin/env python3
"""Coverage audit: where is temporal information observable at all?

For every domain/split we measure
  * queries with >=1 year anchor (query text, or guidance fields),
  * documents with >=1 year mention,
  * gold documents with >=1 year mention,
  * documents with an "ambiguous" time profile (multiple distant years).

This determines which domains can support a temporal reranking experiment:
if gold documents rarely carry a year, temporal consistency cannot be judged
from text and any temporal module is unobservable there.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_qrels, read_jsonl  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def years_in(text):
    return sorted({int(y) for y in YEAR.findall(text or "")})


def anchors(query, guid):
    parts = [query]
    if guid:
        for k in ("query_summary", "query_temporal_events", "key_time_anchors"):
            v = guid.get(k)
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts.extend(str(x) for x in v)
    return years_in("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="bitcoin economics iota law workplace history politics "
                                        "cardano genealogy hsm monero quant travel")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for d in args.domains.split():
        ddir = os.path.join(ROOT, "data", "track1_tempo", d)
        if not os.path.isdir(ddir):
            continue
        ids, texts = load_corpus(ddir)
        n_doc_year = sum(1 for t in texts if years_in(t))
        qmeta = {r["id"]: r for r in read_jsonl(os.path.join(ddir, f"examples_{args.split}.jsonl"))}
        gpath = os.path.join(ddir, f"guidance_{args.split}.jsonl")
        guid = {r["id"]: r.get("query_guidance", {}) for r in read_jsonl(gpath)} if os.path.isfile(gpath) else {}
        q_with_anchor = sum(1 for qid, m in qmeta.items() if anchors(m["query"], guid.get(qid)))
        q_text_anchor = sum(1 for m in qmeta.values() if years_in(m["query"]))
        qrels = load_qrels(ddir, args.split)
        gold, gold_year = 0, 0
        for q, docs in qrels.items():
            gold += len(docs)
            for did in docs:
                t = texts[ids.index(did)] if did in ids else ""
                if years_in(t):
                    gold_year += 1
        rows.append({
            "domain": d, "docs": len(ids),
            "docs_with_year": n_doc_year,
            "docs_with_year_pct": round(100 * n_doc_year / max(1, len(ids)), 1),
            "dev_queries": len(qmeta),
            "queries_with_anchor": q_with_anchor,
            "queries_with_anchor_pct": round(100 * q_with_anchor / max(1, len(qmeta)), 1),
            "queries_with_year_in_text": q_text_anchor,
            "gold_docs": gold,
            "gold_docs_with_year": gold_year,
            "gold_docs_with_year_pct": round(100 * gold_year / max(1, gold), 1),
        })
        print(f"{d:10s} docs {len(ids):7d} year {rows[-1]['docs_with_year_pct']:5.1f}% "
              f"| queries {len(qmeta):3d} anchored {q_with_anchor:3d} "
              f"({rows[-1]['queries_with_anchor_pct']:4.1f}%) "
              f"| gold {gold:4d} year {rows[-1]['gold_docs_with_year_pct']:5.1f}%", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
