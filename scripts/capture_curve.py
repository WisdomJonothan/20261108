#!/usr/bin/env python3
"""Quantify how much of the in-pool oracle headroom the submitted system captures.

Claim under test
----------------
"the reranker captures only ~5% of the oracle gap inside the frozen top-100"
(results/NEXT_STEPS.md, results/PHASE9_PLAN.md, docs/RESEARCH_LOG.md Sec. E).

Definitions (dev split, official metric = nDCG@10, k = 10)
---------------------------------------------------------
baseline  ``runs/dense_mxbai_<domain>_dev.trec`` -- the mxbai first stage,
          scored exactly as submitted (its own ordering; the run is 1000 deep,
          but nDCG@10 only looks at its top-10, which is inside its top-100).
oracle    ideal re-ordering of the *same 100 documents* that the SFT
          cross-encoder is allowed to re-rank, i.e. the dense top-100
          (``oracle@10_cand100`` in score_run.py's vocabulary).  The system's
          fused top-100 is asserted to be that same set -- it is, for all 519
          dev queries at the frozen configuration -- so system and oracle
          differ only in the ordering of an identical candidate set.
capture   ``(system - baseline) / (oracle - baseline)``.

Aggregation
-----------
The official RETECO score is domain-macro: mean nDCG@10 within a domain, then
mean over the 13 domains.  The headline ``capture`` is therefore the ratio of
those three domain-macro means.  Two alternatives are printed for transparency
but are NOT the official quantity: ``mean_of_domain_captures`` (unweighted mean
of the 13 per-domain ratios, so 3-query iota counts as much as 240-query
history) and ``topic_macro_capture`` (flat mean over all 519 queries, which
over-weights history).  They disagree whenever the denominator is small or
negative in some domain (quant has negative capture: fusion hurts it).

Depth matters
-------------
RESEARCH_LOG.md rule 2: every oracle number must carry its candidate depth.
For that reason a second diagnostic oracle over the full 1000-candidate run is
reported as ``capture_vs_oracle1000``.  It is *not* reachable by a reranker
that only sees 100 candidates; it exists only to show how depth-dependent a
bare "capture %" number is.

Deterministic, CPU only, no GPU, ~5 s for the 13 dev domains (519 queries).

Usage
-----
    python scripts/capture_curve.py
    python scripts/capture_curve.py --json results/capture_curve.json \
        --md results/CAPTURE_CURVE.md

Writes ``results/capture_curve.json`` (per-domain + macro) and appends one
markdown section to ``results/CAPTURE_CURVE.md``.  The append is idempotent:
a section is appended only if an identical table (same sha256) is not already
in the file, so reruns keep the file git-clean.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

ROOT = os.environ.get(
    "RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ir_metrics import ndcg_at_k                      # noqa: E402
from lib_reteco import load_qrels, read_trec_run      # noqa: E402

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]


def oracle_ndcg(candidates, gold, k=10):
    """nDCG@k of the ideal ordering of `candidates` (gold first, order kept)."""
    gold_in = [d for d in candidates if d in gold]
    rest = [d for d in candidates if d not in gold]
    return ndcg_at_k(gold_in + rest, gold, k)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def ratio(num, den, eps=1e-9):
    """Capture ratio, or None when the denominator is degenerate (no headroom)."""
    return num / den if abs(den) > eps else None


def score_domain(domain, data_dir, dense_path, sys_path, k, pool_depth):
    """Per-query nDCG@10 for baseline / oracle / system, plus consistency checks."""
    qrels = load_qrels(os.path.join(data_dir, domain), "dev")
    dense = read_trec_run(dense_path)
    system = read_trec_run(sys_path)

    rows = {}
    pool_identical = 0
    outside_pool = 0
    missing = []
    for qid in sorted(qrels):
        gold = set(qrels[qid])
        if qid not in dense or qid not in system:
            missing.append(qid)
            continue
        d_ranked = [d for _, _, d in dense[qid]]
        s_ranked = [d for _, _, d in system[qid]]
        pool = set(d_ranked[:pool_depth])
        if pool == set(s_ranked[:pool_depth]):
            pool_identical += 1
        else:                                  # safety net: widen, never lose gold
            pool |= set(s_ranked[:pool_depth])
        top10 = set(s_ranked[:k])
        outside_pool += len(top10 - pool)

        rows[qid] = {
            "dense": ndcg_at_k(d_ranked, gold, k),
            "oracle100": oracle_ndcg(d_ranked[:pool_depth], gold, k),
            "oracle_all": oracle_ndcg(d_ranked, gold, k),
            "system": ndcg_at_k(s_ranked, gold, k),
        }

    caps, caps_all = [], []
    for r in rows.values():
        c = ratio(r["system"] - r["dense"], r["oracle100"] - r["dense"])
        if c is not None:
            caps.append(c)
        c = ratio(r["system"] - r["dense"], r["oracle_all"] - r["dense"])
        if c is not None:
            caps_all.append(c)

    n = len(rows)
    out = {
        "domain": domain,
        "num_queries": len(qrels),
        "num_scored": n,
        "topics_missing_from_run": missing,
        "dense_ndcg10": mean(r["dense"] for r in rows.values()),
        "oracle_ndcg10_cand100": mean(r["oracle100"] for r in rows.values()),
        "oracle_ndcg10_cand1000": mean(r["oracle_all"] for r in rows.values()),
        "system_ndcg10": mean(r["system"] for r in rows.values()),
    }
    out["oracle_gain_cand100"] = (out["oracle_ndcg10_cand100"] - out["dense_ndcg10"])
    out["delta_vs_dense"] = out["system_ndcg10"] - out["dense_ndcg10"]
    out["capture"] = ratio(out["delta_vs_dense"], out["oracle_gain_cand100"])
    out["capture_vs_oracle1000"] = ratio(
        out["delta_vs_dense"],
        out["oracle_ndcg10_cand1000"] - out["dense_ndcg10"])
    out["per_query"] = {
        "capture_mean": mean(caps),
        "capture_median": sorted(caps)[len(caps) // 2] if caps else None,
        "frac_queries_improved": (sum(1 for r in rows.values()
                                      if r["system"] > r["dense"] + 1e-12) / n) if n else None,
        "frac_queries_worsened": (sum(1 for r in rows.values()
                                      if r["system"] < r["dense"] - 1e-12) / n) if n else None,
        "n_with_headroom": len(caps),
    }
    checks = {"pool_identical": pool_identical, "system_top10_outside_pool": outside_pool}
    return out, checks, rows


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(x):
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def build_markdown(rep):
    m = rep["macro"]
    lines = []
    lines.append("| domain | n | dense nDCG@10 | oracle nDCG@10 (top-100) | system nDCG@10 "
                 "| delta system-dense | capture |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in rep["domains"]:
        lines.append(
            f"| {d['domain']} | {d['num_scored']} | {fmt(d['dense_ndcg10'])} | "
            f"{fmt(d['oracle_ndcg10_cand100'])} | {fmt(d['system_ndcg10'])} | "
            f"{d['delta_vs_dense']:+.4f} | {pct(d['capture'])} |")
    lines.append(
        f"| **domain-macro (official)** | {m['num_queries']} | {fmt(m['dense_ndcg10'])} | "
        f"{fmt(m['oracle_ndcg10_cand100'])} | {fmt(m['system_ndcg10'])} | "
        f"{m['delta_vs_dense']:+.4f} | **{pct(m['capture'])}** |")
    return "\n".join(lines)


def append_markdown(path, rep, table):
    body_hash = hashlib.sha256(table.encode("utf-8")).hexdigest()[:16]
    marker = f"<!-- capture-curve:{body_hash} -->"
    existing = ""
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            existing = f.read()
    if marker in existing:
        print(f"[capture_curve] identical table already in {path} ({marker}); not appending")
        return False

    m = rep["macro"]
    section = [
        "",
        f"## {datetime.date.today().isoformat()} - capture curve ({body_hash})",
        "",
        f"Runs: `{rep['meta']['dense_pattern']}` vs `{rep['meta']['system_pattern']}`, "
        f"dev split, k={rep['meta']['k']}, pool depth={rep['meta']['pool_depth']}.",
        "",
        table,
        "",
        f"- baseline = dense mxbai first stage; oracle = ideal ordering of the frozen "
        f"top-100; capture = (system - baseline) / (oracle - baseline), computed on the "
        f"three domain-macro means.",
        f"- unweighted mean of the 13 per-domain capture ratios = {pct(m['mean_of_domain_captures'])} "
        f"(gives a 3-query domain the same weight as 240-query history).",
        f"- flat topic-macro capture over all {m['num_queries']} queries = "
        f"{pct(m['topic_macro_capture'])}.",
        f"- depth sensitivity: capture against the 1000-deep oracle (not reachable by a "
        f"top-100 reranker) = {pct(m['capture_vs_oracle1000'])}.",
        f"- per-query capture: mean {pct(m['per_query_capture_mean'])}, "
        f"median {pct(m['per_query_capture_median'])}; "
        f"{pct(m['frac_queries_improved'])} of queries improve, "
        f"{pct(m['frac_queries_worsened'])} get worse.",
        marker,
        "",
    ]
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        if not existing:
            f.write("# Capture curve: how much of the frozen-top-100 oracle gap the "
                    "submitted system actually captures\n")
        f.write("\n".join(section))
    print(f"[capture_curve] appended section {body_hash} to {path}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--data-dir", default="data/track1_tempo")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--dense-pattern", default="runs/dense_mxbai_{domain}_dev.trec")
    ap.add_argument("--system-pattern", default="runs/sub_sft11_{domain}_dev.trec")
    ap.add_argument("--json", default="results/capture_curve.json")
    ap.add_argument("--md", default="results/CAPTURE_CURVE.md")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool-depth", type=int, default=100)
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    data_dir = os.path.join(root, args.data_dir)

    domains, checks = [], {"pool_identical": 0, "system_top10_outside_pool": 0,
                           "topics_missing": 0, "num_queries": 0}
    all_rows = []
    for d in DOMAINS:
        dense_path = os.path.join(root, args.dense_pattern.format(domain=d))
        sys_path = os.path.join(root, args.system_pattern.format(domain=d))
        for p in (dense_path, sys_path):
            if not os.path.isfile(p):
                raise SystemExit(f"missing run file: {p}")
        row, ck, qrows = score_domain(d, data_dir, dense_path, sys_path,
                                      args.k, args.pool_depth)
        for qid, r in qrows.items():
            all_rows.append({"domain": d, "qid": qid, **r})
        domains.append(row)
        checks["pool_identical"] += ck["pool_identical"]
        checks["system_top10_outside_pool"] += ck["system_top10_outside_pool"]
        checks["topics_missing"] += len(row["topics_missing_from_run"])
        checks["num_queries"] += row["num_queries"]

    def macro(key):
        vals = [d[key] for d in domains if d[key] is not None]
        return sum(vals) / len(vals) if vals else None

    dm = {
        "num_domains": len(domains),
        "num_queries": checks["num_queries"],
        "dense_ndcg10": macro("dense_ndcg10"),
        "oracle_ndcg10_cand100": macro("oracle_ndcg10_cand100"),
        "oracle_ndcg10_cand1000": macro("oracle_ndcg10_cand1000"),
        "system_ndcg10": macro("system_ndcg10"),
    }
    dm["oracle_gain_cand100"] = dm["oracle_ndcg10_cand100"] - dm["dense_ndcg10"]
    dm["delta_vs_dense"] = dm["system_ndcg10"] - dm["dense_ndcg10"]
    dm["capture"] = ratio(dm["delta_vs_dense"], dm["oracle_gain_cand100"])
    dm["capture_vs_oracle1000"] = ratio(
        dm["delta_vs_dense"],
        dm["oracle_ndcg10_cand1000"] - dm["dense_ndcg10"])
    dm["mean_of_domain_captures"] = mean(d["capture"] for d in domains)

    # Flat (topic-macro) statistics over all 519 queries -- computed directly from the
    # per-query rows, not by averaging the 13 domain means (that would be domain-macro).
    n_all = len(all_rows)
    sums = {k: sum(r[k] for r in all_rows) for k in ("dense", "oracle100", "system")}
    dm["topic_macro_capture"] = ratio(
        (sums["system"] - sums["dense"]) / n_all,
        (sums["oracle100"] - sums["dense"]) / n_all)
    pq = [ratio(r["system"] - r["dense"], r["oracle100"] - r["dense"]) for r in all_rows]
    pq = sorted(c for c in pq if c is not None)
    dm["per_query_capture_mean"] = sum(pq) / len(pq) if pq else None
    dm["per_query_capture_median"] = pq[len(pq) // 2] if pq else None
    dm["frac_queries_improved"] = (sum(1 for r in all_rows
                                       if r["system"] > r["dense"] + 1e-12) / n_all) if n_all else None
    dm["frac_queries_worsened"] = (sum(1 for r in all_rows
                                       if r["system"] < r["dense"] - 1e-12) / n_all) if n_all else None
    dm["num_queries_with_headroom"] = len(pq)

    rep = {
        "meta": {
            "data_dir": args.data_dir, "split": args.split, "k": args.k,
            "pool_depth": args.pool_depth, "domains": DOMAINS,
            "dense_pattern": args.dense_pattern, "system_pattern": args.system_pattern,
            "generated": datetime.date.today().isoformat(),
        },
        "definition": {
            "baseline": "dense_mxbai first stage, scored as submitted (nDCG@10 of its own ordering)",
            "oracle": "ideal re-ordering of the dense top-100 (oracle@10_cand100); the "
                      "system's fused top-100 is asserted to be the same set",
            "capture": "(system_ndcg10 - baseline) / (oracle_ndcg10_cand100 - baseline)",
            "aggregation": "ratio of domain-macro means (official metric); alternatives reported "
                           "separately as mean_of_domain_captures and topic_macro_capture",
            "note": "capture_vs_oracle1000 is a depth-sensitivity diagnostic, not a reachable ceiling",
        },
        "domains": domains,
        "macro": dm,
        "checks": {
            "num_queries": checks["num_queries"],
            "topics_with_identical_top100_pool": checks["pool_identical"],
            "system_top10_outside_pool": checks["system_top10_outside_pool"],
            "topics_missing_from_run": checks["topics_missing"],
        },
    }

    json_path = os.path.join(root, args.json)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
        f.write("\n")

    table = build_markdown(rep)
    print(f"# capture curve ({args.dense_pattern} -> {args.system_pattern})")
    print(table)
    print()
    print(f"pool check: {checks['pool_identical']}/{checks['num_queries']} topics have an "
          f"identical dense/system top-100; system top-10 docs outside the pool: "
          f"{checks['system_top10_outside_pool']}; missing topics: {checks['topics_missing']}")
    print(f"domain-macro capture = {pct(dm['capture'])}  "
          f"(mean of per-domain ratios {pct(dm['mean_of_domain_captures'])}, "
          f"vs 1000-deep oracle {pct(dm['capture_vs_oracle1000'])})")
    print(f"-> {json_path}")

    if not args.no_append:
        append_markdown(os.path.join(root, args.md), rep, table)


if __name__ == "__main__":
    main()
