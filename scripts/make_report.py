#!/usr/bin/env python3
"""Assemble the Phase-0 report from the saved result JSONs (no recomputation)."""
import glob
import json
import os

R = "D:/RETECO/results"
PUBLISHED = {("iota", "dev"): 0.3309, ("iota", "train"): 0.1563,
             ("bitcoin", "dev"): 0.0370, ("bitcoin", "train"): 0.0856}


def load(name):
    p = os.path.join(R, name)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


def main():
    env = load("env.json")
    lines = []
    w = lines.append

    w("# RETECO Track 1a — Phase 0 report (IOTA / Bitcoin)\n")
    w("All numbers below were produced on this machine from the official v1.1 "
      "release and the official Pyserini prebuilt indexes; none are quoted from "
      "the organizers except the `published` column used for verification.\n")

    if env:
        w("## Environment\n")
        w(f"- Python: `{env['python'].split('|')[0].strip()}` ({env['executable']})")
        w(f"- Java: `{env['java'].splitlines()[0]}`")
        pk = env["packages"]
        w("- Packages: " + ", ".join(f"{k} {v}" for k, v in pk.items() if v))
        if isinstance(env.get("cuda"), dict):
            w(f"- GPU: {env['cuda'].get('device')} (torch CUDA {env['cuda'].get('torch_cuda')})")
        w("")

    w("## BM25: official starter-kit baseline vs official prebuilt Lucene index\n")
    w("| domain | split | topics | official nDCG@10 | published | prebuilt index nDCG@10 | R@100 off | R@100 idx | oracle@10 idx (cand100) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for dom in ("iota", "bitcoin"):
        for split in ("dev", "train"):
            o = load(f"score_bm25_official_{dom}_{split}.json")
            p = load(f"score_bm25_pyserini_{dom}_{split}.json")
            if not (o and p):
                continue
            w(f"| {dom} | {split} | {o['num_topics']} | {fmt(o['nDCG@10'])} | "
              f"{PUBLISHED[(dom, split)]:.4f} | {fmt(p['nDCG@10'])} | "
              f"{fmt(o['Recall@100'])} | {fmt(p['Recall@100'])} | "
              f"{fmt(p['oracle_nDCG@10_cand100'])} |")
    w("")

    w("## Dense (zero-shot bge-base-en-v1.5) vs BM25 vs candidate union\n")
    w("| domain | split | BM25 nDCG@10 | Dense nDCG@10 | Dense R@100 | Union R@100 | BM25 R@100 | oracle@10 union (cand100) |")
    w("|---|---|---|---|---|---|---|---|")
    for dom in ("iota", "bitcoin"):
        for split in ("dev", "train"):
            b = load(f"score_bm25_pyserini_{dom}_{split}.json")
            d = load(f"score_dense_bge_{dom}_{split}.json")
            u = load(f"union_{dom}_{split}.json")
            if not (b and d):
                continue
            w(f"| {dom} | {split} | {fmt(b['nDCG@10'])} | {fmt(d['nDCG@10'])} | "
              f"{fmt(d['Recall@100'])} | "
              f"{fmt(u['Recall@100_Union']) if u else 'n/a'} | {fmt(b['Recall@100'])} | "
              f"{fmt(u['oracle_nDCG@10_Union_cand100']) if u else 'n/a'} |")
    w("")

    w("## Candidate-depth sweep (recall and candidate-oracle at the same depth)\n")
    w("`oracle` = ideal re-ranking of exactly the top-k candidates, so "
      "`oracle == 0` whenever `recall == 0` (asserted for every query). "
      "`union` uses a 2k budget (diagnostic); `RRF@k` is the equal-budget fusion, "
      "fused from the full lists and truncated to k, scored for real.\n")
    w("| domain | split | k | BM25 rec | BM25 orc | dense rec | dense orc | union rec | union orc | RRF nDCG@10 | RRF rec |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for dom in ("iota", "bitcoin"):
        for split in ("dev", "train"):
            s = load(f"sweep_{dom}_{split}.json")
            if not s:
                continue
            for k in ("20", "50", "100", "200", "1000"):
                r = s["by_k"].get(k)
                if not r:
                    continue
                w(f"| {dom} | {split} | {k} | {fmt(r['Recall@k_A'])} | {fmt(r['oracle_nDCG@10_A'])} | "
                  f"{fmt(r['Recall@k_B'])} | {fmt(r['oracle_nDCG@10_B'])} | "
                  f"{fmt(r['Recall@k_Union'])} | {fmt(r['oracle_nDCG@10_Union'])} | "
                  f"{fmt(r['RRF_nDCG@10'])} | {fmt(r['Recall@k_RRF'])} |")
    w("")

    w("## Temporal feasibility (content-time signals in the released corpus)\n")
    w("| domain | docs | with 4-digit year | ISO date | month name | temporal-flagged queries | gold docs with a year |")
    w("|---|---|---|---|---|---|---|")
    for dom in ("iota", "bitcoin"):
        t = load(f"time_diag_{dom}_dev.json")
        if not t:
            continue
        w(f"| {dom} | {t['num_docs']} | {t['pct_any_year']:.1f}% | {t['pct_iso_date']:.1f}% | "
          f"{t['pct_month_name']:.1f}% | {t['num_queries_flagged_temporal']}/{t['num_queries']} | "
          f"{100*t['mean_gold_with_year_frac']:.1f}% |")
    w("")

    w("## Per-query comparison: dense vs BM25 (same candidate budget, top-1000)\n")
    w("| domain | split | queries | dense wins | ties | BM25 wins | BM25 nDCG@10=0 | dense nDCG@10=0 | BM25 no gold@100 | dense no gold@100 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for dom in ("iota", "bitcoin"):
        for split in ("dev", "train"):
            b = load(f"score_bm25_pyserini_{dom}_{split}.json")
            d = load(f"score_dense_bge_{dom}_{split}.json")
            if not (b and d):
                continue
            bq, dq = b["per_query"], d["per_query"]
            qs = [q for q in bq if q in dq]
            win = sum(1 for q in qs if dq[q]["nDCG@10"] > bq[q]["nDCG@10"] + 1e-12)
            lose = sum(1 for q in qs if dq[q]["nDCG@10"] < bq[q]["nDCG@10"] - 1e-12)
            tie = len(qs) - win - lose
            w(f"| {dom} | {split} | {len(qs)} | {win} | {tie} | {lose} | "
              f"{sum(1 for q in qs if bq[q]['nDCG@10'] == 0)} | "
              f"{sum(1 for q in qs if dq[q]['nDCG@10'] == 0)} | "
              f"{sum(1 for q in qs if bq[q]['Rec@100'] == 0)} | "
              f"{sum(1 for q in qs if dq[q]['Rec@100'] == 0)} |")
    w("")

    w("## Cross-check: official `scorer.py` on the same run files\n")
    w("| run | our nDCG@10 | official scorer nDCG@10 | |diff| | TP diff | TC diff |")
    w("|---|---|---|---|---|---|")
    worst = 0.0
    for dom in ("iota", "bitcoin"):
        for split in ("dev", "train"):
            for sysname, tag in (("bm25_official", "bm25_official"),
                                 ("bm25_pyserini", "bm25_pyserini"),
                                 ("dense_bge", "dense_bge")):
                ours = load(f"score_{sysname}_{dom}_{split}.json")
                off = load(f"official_scorer_{sysname}_{dom}_{split}.json")
                if not (ours and off):
                    continue
                d1 = abs(ours["nDCG@10"] - off["nDCG@10"])
                d2 = abs(ours["TP@10"] - off["TP@10"])
                d3 = abs(ours["TC@10"] - off["TC@10"])
                worst = max(worst, d1, d2, d3)
                w(f"| {sysname} {dom} {split} | {ours['nDCG@10']:.4f} | "
                  f"{off['nDCG@10']:.4f} | {d1:.4f} | {d2:.4f} | {d3:.4f} |")
    w("")
    w(f"Maximum absolute difference across all 12 runs: **{worst:.6f}** "
      "(the scoring path used for the tables above is therefore the official metric).")
    w("")

    lat = load("latency_iota.json")
    if lat:
        w("## Retrieval latency (warm, IOTA index, 3 repetitions per query)\n")
        for k, v in lat["k"].items():
            w(f"- top-{k}: mean {v['mean_ms']:.1f} ms, p50 {v['p50_ms']:.1f} ms")
        w("")

    out = os.path.join(R, "PHASE0_REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
