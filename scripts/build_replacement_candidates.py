#!/usr/bin/env python3
"""Frozen candidate + comparison list for the bounded one-swap replacement study.

Protocol (fixed before looking at dev labels; candidate generation never reads qrels):

  1. Candidates for query q: (dense top-20) UNION (full-query CE top-20), minus the
     frozen system's top-10 documents.
  2. Order candidates by their BEST rank in either source (DENSE_ORDER[i] = i);
     tie broken by source preference dense-before-CE, then by docid (lexicographic).
  3. Keep at most MAX_CANDIDATES = 3.
  4. Comparison pairs: every kept candidate against frozen ranks 8, 9 and 10
     (<= 3 x 3 = 9 pairs per query).
  5. A policy may execute at most ONE swap per query; the rest of the ranking stays
     exactly as frozen.

This script only *builds* the lists (and dumps the ranking features that later
policies/gates are allowed to see).  Oracle computation lives in
eval_replacement_pilot.py --mode oracle-fixed so that the method and the oracle
provably consume the same candidate file.

Outputs (under results/replacement_v2/):
  candidates_<split>[_<ce tag>].json   frozen candidate + comparison list
  candidates_<split>[_<ce tag>].txt    human-readable protocol summary
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]

MAX_CANDIDATES = 3
SLOTS = (7, 8, 9)          # 0-based -> ranks 8, 9, 10
TOP_SOURCE_DEPTH = 20
FROZEN_TOP = 10
DENSE_ORDER = {"dense": 0, "ce": 1}   # tie rule: dense wins


def read_trec_run(path):
    """{qid: [(rank, score, docid)]} sorted by rank."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            qid, _q0, docid, rank, score, _tag = ln.split()
            out.setdefault(qid, []).append((int(rank), float(score), docid))
    for qid in out:
        out[qid].sort(key=lambda r: r[0])
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zscores(run_q):
    """Per-query z-normalised CE score (deployment convention, w=1)."""
    vals = [s for _, s, _ in run_q]
    n = len(vals)
    if n < 2:
        return {d: 0.0 for _, _, d in run_q}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    sd = var ** 0.5 or 1e-8
    return {d: (s - mean) / sd for _, s, d in run_q}


QIDS = None


def build_domain(root, domain, split, frozen_tag, dense_tag, ce_tag, ce_dir):
    data = Path(root) / "data/track1_tempo" / domain
    frozen = read_trec_run(Path(root) / "runs" / f"{frozen_tag}_{domain}_{split}.trec")
    dense = read_trec_run(Path(root) / "runs" / f"{dense_tag}_{domain}_{split}.trec")
    ce_path = Path(ce_dir) / f"{ce_tag}_{domain}_{split}_ce.trec"
    if not ce_path.exists():
        ce_path = Path(root) / "runs" / f"{ce_tag}_{domain}_{split}_ce.trec"
    ce = read_trec_run(ce_path)

    queries = {}
    with (data / f"examples_{split}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            queries[item["id"]] = item["query"]
    if QIDS is not None:
        queries = {q: t for q, t in queries.items() if q in QIDS}
    assert set(queries) == set(frozen), (domain, len(queries), len(frozen))
    assert set(ce) <= set(dense), (domain, "ce query not in dense run")

    rows = []
    for q in sorted(queries):
        top = [d for _, _, d in frozen[q][:FROZEN_TOP]]
        assert len(top) == FROZEN_TOP
        # ranking features the policy is allowed to use
        dense_rank = {d: r for r, _, d in dense[q][:TOP_SOURCE_DEPTH]}
        ce_rank = {d: r for r, _, d in ce[q][:TOP_SOURCE_DEPTH]}
        dense_score = {d: s for _, s, d in dense[q]}
        ce_z = zscores(ce[q])
        in_frozen = set(d for _, _, d in frozen[q])

        pool = {}
        for d, r in dense_rank.items():
            pool.setdefault(d, {})["dense_rank"] = r
        for d, r in ce_rank.items():
            pool.setdefault(d, {})["ce_rank"] = r
        cands = []
        for d, rec in pool.items():
            if d in top:
                continue
            best = min(rec.get("dense_rank", 10 ** 6), rec.get("ce_rank", 10 ** 6))
            # which source supplied the best rank (dense preferred on ties)
            if rec.get("dense_rank", 10 ** 6) <= rec.get("ce_rank", 10 ** 6):
                src = "dense"
            else:
                src = "ce"
            cands.append({
                "docid": d, "best_rank": best, "best_source": src,
                "dense_rank": rec.get("dense_rank"), "ce_rank": rec.get("ce_rank"),
                "frozen_rank": (frozen[q].index(next(x for x in frozen[q] if x[2] == d)) + 1
                                if d in in_frozen else None),
                "dense_score": dense_score.get(d),
                "ce_z": ce_z.get(d),
            })
        cands.sort(key=lambda c: (c["best_rank"], DENSE_ORDER[c["best_source"]], c["docid"]))
        kept = cands[:MAX_CANDIDATES]

        pairs = []
        for slot in SLOTS:
            slot_doc = top[slot]
            for c in kept:
                pairs.append({
                    "slot_rank": slot + 1,
                    "removed_docid": slot_doc,
                    "incoming_docid": c["docid"],
                    "incoming_best_rank": c["best_rank"],
                    "incoming_best_source": c["best_source"],
                    "incoming_dense_rank": c["dense_rank"],
                    "incoming_ce_rank": c["ce_rank"],
                    "removed_frozen_rank": slot + 1,
                    "removed_dense_rank": dense_rank.get(slot_doc),
                    "removed_ce_rank": ce_rank.get(slot_doc),
                    "incoming_dense_score": c["dense_score"],
                    "incoming_ce_z": c["ce_z"],
                    "removed_dense_score": dense_score.get(slot_doc),
                    "removed_ce_z": ce_z.get(slot_doc),
                    "incoming_dense_gap": (c["dense_score"] - dense_score[slot_doc])
                    if (c["dense_score"] is not None and slot_doc in dense_score) else None,
                    "incoming_ce_gap": (c["ce_z"] - ce_z[slot_doc])
                    if slot_doc in ce_z else None,
                })
        rows.append({
            "qid": q, "domain": domain, "query": queries[q],
            "frozen_top10": top,
            "n_pool_docs": len(pool), "n_candidates_total": len(cands),
            "candidates": kept,
            "pairs": pairs,
        })
    return rows, ce_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--frozen-tag", default="sub_sft_e2bc")
    ap.add_argument("--dense-tag", default="dense_mxbai")
    ap.add_argument("--ce-tag", default="fusion100_full_ce_s17_20261115")
    ap.add_argument("--ce-dir", default="runs")
    ap.add_argument("--qids-file", default="")
    ap.add_argument("--out-tag", default="candidates")
    ap.add_argument("--out", default="results/replacement_v2")
    args = ap.parse_args()
    root = args.root
    if args.qids_file:
        import json as _json
        global QIDS
        QIDS = set(str(q) for q in _json.loads((root / args.qids_file).read_text(encoding="utf-8")))
        print("qid filter:", len(QIDS))
    outdir = root / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    rows, inputs = [], []
    for domain in DOMAINS:
        rows_d, ce_path = build_domain(root, domain, args.split, args.frozen_tag,
                                       args.dense_tag, args.ce_tag, args.ce_dir)
        rows.extend(rows_d)
        inputs.append({
            "domain": domain,
            "frozen": str(root / "runs" / f"{args.frozen_tag}_{domain}_{args.split}.trec"),
            "dense": str(root / "runs" / f"{args.dense_tag}_{domain}_{args.split}.trec"),
            "ce": str(ce_path),
        })
        print(domain, len(rows_d), sum(len(r["candidates"]) for r in rows_d),
              sum(len(r["pairs"]) for r in rows_d), flush=True)

    assert len(rows) > 0, len(rows)
    protocol = {
        "candidate_rule": (f"union(dense top{TOP_SOURCE_DEPTH}, CE top{TOP_SOURCE_DEPTH}) "
                           f"minus frozen top{FROZEN_TOP}; order by best rank "
                           "(tie: dense before CE, then docid); keep top "
                           f"{MAX_CANDIDATES}"),
        "comparison_rule": "each candidate vs frozen ranks 8,9,10 (<= 9 pairs)",
        "swap_rule": "at most one swap per query, other ranks frozen",
        "labels_used_for_candidates": False,
        "slot_ranks": [s + 1 for s in SLOTS],
        "max_candidates": MAX_CANDIDATES,
        "frozen_tag": args.frozen_tag,
        "dense_tag": args.dense_tag,
        "ce_tag": args.ce_tag,
        "split": args.split,
        "zscore_convention": "per-query z over the CE run file (deployment w=1)",
    }
    payload = {"protocol": protocol, "inputs": inputs,
               "n_queries": len(rows),
               "n_pairs": sum(len(r["pairs"]) for r in rows),
               "queries_with_3_candidates": sum(len(r["candidates"]) == MAX_CANDIDATES for r in rows),
               "queries": rows}
    if args.out_tag and args.out_tag != "candidates":
        name = f"{args.out_tag}.json"
    else:
        name = f"candidates_{args.split}" + ("" if args.ce_tag == "fusion100_full_ce_s17_20261115"
                                             else f"_{args.ce_tag}") + ".json"
    path = outdir / name
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    summary = [f"file: {path}", f"protocol: {json.dumps(protocol, indent=1)}", ""]
    for dom in DOMAINS:
        sub = [r for r in rows if r["domain"] == dom]
        summary.append(f"{dom:10s} q={len(sub):3d} pairs={sum(len(r['pairs']) for r in sub):4d} "
                       f"cand3={sum(len(r['candidates'])==3 for r in sub):3d}")
    summary += [
        "", f"queries={len(rows)} pairs={payload['n_pairs']} "
            f"queries_with_3_candidates={payload['queries_with_3_candidates']}",
        "input sha256:",
    ] + [f"  {x['domain']}: frozen={sha256(x['frozen'])[:16]} dense={sha256(x['dense'])[:16]} "
         f"ce={sha256(x['ce'])[:16]}" for x in inputs]
    (outdir / name.replace(".json", ".txt")).write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"n_queries": payload["n_queries"], "n_pairs": payload["n_pairs"],
                      "queries_with_3_candidates": payload["queries_with_3_candidates"],
                      "out": str(path)}))


if __name__ == "__main__":
    main()
