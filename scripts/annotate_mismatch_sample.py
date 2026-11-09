#!/usr/bin/env python3
"""Build a blinded coding sheet of year-mismatched (query, gold) pairs.

Frame: dev queries, frozen dense mxbai top-100, gold documents whose extracted
year does not overlap the query interval (gap > 0).  The sheet hides the domain
and the docid so that the annotator codes the *mechanism* without being able to
look the document up or infer relevance from the id (the shared task warned that
ids correlate with labels).  A separate key file keeps the provenance.

    python scripts/annotate_mismatch_sample.py --out results/mismatch

Outputs
  results/mismatch/annotator_sheet.jsonl  (blinded items, mechanism left empty)
  results/mismatch/key.json               (item_id -> domain/qid/docid/gap)
  results/mismatch/frame_stats.json       (how many pairs exist per domain)
"""
import argparse
import hashlib
import html
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_reteco import load_corpus, load_examples, load_qrels, read_trec_run  # noqa: E402
from hardneg_mine import doc_time, load_guidance, query_interval, gap  # noqa: E402

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]
HTMLTAG = re.compile(r"<[^>]+>")


def clean(text):
    text = HTMLTAG.sub(" ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--snippet", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mismatch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    items, key, stats = [], {}, {}
    for dom in DOMAINS:
        ddir = os.path.join(ROOT, "data", "track1_tempo", dom)
        run_path = os.path.join(ROOT, "runs", f"dense_mxbai_{dom}_{args.split}.trec")
        if not os.path.isfile(run_path):
            stats[dom] = {"skip": "no_run"}
            continue
        ids, docs = load_corpus(ddir)
        docs_by_id = dict(zip(ids, docs))
        qids, queries = load_examples(ddir, args.split)
        gold = {q: set(d) for q, d in load_qrels(ddir, args.split).items()}
        guid = load_guidance(ddir, args.split)
        run = read_trec_run(run_path)
        cand = []
        for qid in qids:
            iv = query_interval(queries[qids.index(qid)], guid.get(qid, {}))
            if iv is None:
                continue
            for _r, _s, d in run.get(qid, [])[: args.k]:
                if d not in gold.get(qid, set()):
                    continue
                dt = doc_time(docs_by_id.get(d, ""))
                if dt is None:
                    continue
                g = gap(dt["years"], iv)
                if g > 0:
                    cand.append({"domain": dom, "qid": qid, "docid": d,
                                 "gap": g, "years": dt["years"],
                                 "conf": dt.get("conf"), "iv": list(iv)})
        stats[dom] = {"mismatched_gold": len(cand),
                      "gold": sum(len(v) for v in gold.values())}
        rng.shuffle(cand)
        for c in cand[: args.per_domain]:
            item_id = hashlib.sha1(f"{c['domain']}|{c['qid']}|{c['docid']}|ann".encode()).hexdigest()[:12]
            qtext = clean(queries[qids.index(c["qid"])])
            g = guid.get(c["qid"], {})
            sheet = {
                "item_id": item_id,
                "query": qtext[:600],
                "query_summary": clean(g.get("query_summary", ""))[:400],
                "query_time_anchors": g.get("key_time_anchors"),
                "query_years_seen": list(c["iv"]),
                "doc_years_seen": c["years"],
                "doc_snippet": clean(docs_by_id.get(c["docid"], ""))[: args.snippet],
                "mechanism": "",
                "evidence": "",
                "confidence": None,
            }
            items.append(sheet)
            key[item_id] = {"domain": c["domain"], "qid": c["qid"],
                            "docid": c["docid"], "gap": c["gap"]}

    rng.shuffle(items)
    with open(os.path.join(args.out, "annotator_sheet.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    json.dump(key, open(os.path.join(args.out, "key.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"frame": stats, "items": len(items)}, 
              open(os.path.join(args.out, "frame_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"items={len(items)} -> {args.out}/annotator_sheet.jsonl")
    tot = sum(v.get("mismatched_gold", 0) for v in stats.values())
    print(f"frame: {tot} mismatched gold pairs over {len(stats)} domains")


if __name__ == "__main__":
    main()
