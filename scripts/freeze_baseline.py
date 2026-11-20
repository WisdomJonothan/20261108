#!/usr/bin/env python3
"""Freeze + re-verify the 0.3581 dev baseline (step 1 of the research plan).

Produces results/FROZEN_BASELINE_20261120.json and .md:
  * every component of the frozen system pinned by hash (checkpoint, the 13 dev
    run files, dense/CE inputs, evaluation scripts, data files),
  * an independent re-score of the frozen run files with the official metric,
  * a coverage audit (519 queries / 13 domains, nothing silently dropped),
  * the old recorded number next to the new re-run number.

Nothing here runs a model; it only re-scores stored run files and hashes inputs.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]

CONFIG = {
    "frozen_tag": "sub_sft_e2bc",
    "description": "dense (mxbai-embed-large-v1, top-100) + SFT cross-encoder (title query, 2 epochs, lr 1.5e-5), z-normalised 1:1 fusion, detied",
    "dense": {"model": "mixedbread-ai/mxbai-embed-large-v1", "pooling": "mean",
              "query_prefix": "Represent this sentence for searching relevant passages: ",
              "top_k": 100},
    "reranker": {"base": "BAAI/bge-reranker-v2-m3", "checkpoint": "results/ce_ft/v2m3_pool8_e2/best",
                 "query_mode": "title", "query_max_tokens": 256,
                 "window_tokens": 250, "stride_tokens": 192, "max_windows_per_doc": 3,
                 "validation_pair_max_len": 320, "deployment_pair_max_len": 510},
    "fusion": {"normalisation": "z", "weight_dense": 1.0, "weight_ce": 1.0, "tie_breaking": "detie"},
    "evaluation": {"split": "dev", "metric": "nDCG@10", "aggregation": "domain-macro (13 domains)",
                   "scorer": "scripts/ir_metrics.py (vendored official)", "k": 10},
    "data": {"corpus": "track1_tempo v1.1", "qrels": "data/track1_tempo/<domain>/qrels_dev.txt",
             "queries": "data/track1_tempo/<domain>/examples_dev.jsonl"},
}


def sha256(path, chunk=1 << 20):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dir_fingerprint(path):
    p = Path(path)
    if not p.exists():
        return None
    out = {}
    for f in sorted(p.rglob("*")):
        if f.is_file():
            out[str(f.relative_to(p))] = sha256(f)
    return out


def rescore(root, python, tag):
    tmp = root / "results/verify20261120"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DOMAINS:
        js = tmp / f"{tag}_{d}_dev.json"
        if not js.exists():
            subprocess.run([python, str(root / "scripts/score_run.py"),
                            "--run", str(root / "runs" / f"{tag}_{d}_dev.trec"),
                            "--data", str(root / "data/track1_tempo" / d),
                            "--split", "dev", "--json", str(js), "--tag", tag],
                           check=True, capture_output=True)
        rows.append(json.loads(js.read_text(encoding="utf-8")))
    agg = tmp / f"aggregate_{tag}_dev.json"
    subprocess.run([python, str(root / "scripts/aggregate_scores.py"),
                    "--pattern", str(tmp / f"{tag}_*_dev.json"), "--tag", tag,
                    "--metric", "ndcg@10", "--json", str(agg)],
                   check=True, capture_output=True)
    return rows, json.loads(agg.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--old-value", type=float, default=0.35809852326897723)
    ap.add_argument("--out", default="results/FROZEN_BASELINE_20261120")
    args = ap.parse_args()
    root = args.root
    python = sys.executable
    tag = CONFIG["frozen_tag"]
    rows, agg = rescore(root, python, tag)

    coverage = []
    for r in rows:
        domain = Path(r["data"]).name
        coverage.append({
            "domain": domain, "num_topics": r["num_topics"],
            "num_topics_in_run": r["num_topics_in_run"],
            "missing_from_run": len(r["topics_missing_from_run"]),
            "nDCG@10": r["nDCG@10"],
            "qrels_queries": sum(1 for ln in (Path(r["data"]) / "qrels_dev.txt")
                                 .read_text(encoding="utf-8").splitlines() if ln.strip()),
        })
    total_topics = sum(c["num_topics"] for c in coverage)
    total_qrels_queries = len({ln.split()[0] for c in coverage
                               for ln in (root / "data/track1_tempo" / c["domain"] / "qrels_dev.txt")
                               .read_text(encoding="utf-8").splitlines() if ln.strip()})

    inputs = {
        "dev_runs": {d: sha256(root / "runs" / f"{tag}_{d}_dev.trec") for d in DOMAINS},
        "dense_runs": {d: sha256(root / "runs" / f"dense_mxbai_{d}_dev.trec") for d in DOMAINS},
        "qrels_dev": {d: sha256(root / "data/track1_tempo" / d / "qrels_dev.txt") for d in DOMAINS},
        "examples_dev": {d: sha256(root / "data/track1_tempo" / d / "examples_dev.jsonl") for d in DOMAINS},
        "eval_scripts": {s: sha256(root / "scripts" / s) for s in
                         ("score_run.py", "aggregate_scores.py", "ir_metrics.py",
                          "bootstrap_compare.py", "lib_reteco.py")},
        "checkpoint": dir_fingerprint(root / CONFIG["reranker"]["checkpoint"]),
    }
    baseline = {
        "created": "2026-11-20",
        "config": CONFIG,
        "old_recorded": {
            "file": "results/aggregate_sub_sft_e2bc_dev.json",
            "domain_macro": args.old_value,
            "topic_macro": 0.36436669567697244,
        },
        "new_reverification": {
            "command": "scripts/score_run.py (13 domains) -> scripts/aggregate_scores.py --metric ndcg@10",
            "domain_macro": agg["domain_macro"],
            "topic_macro": agg["topic_macro"],
            "num_domains": agg["num_domains"],
            "num_topics": agg["num_topics"],
            "delta_vs_old": agg["domain_macro"] - args.old_value,
            "identical": abs(agg["domain_macro"] - args.old_value) < 1e-12,
        },
        "coverage_audit": {
            "domains": len(coverage),
            "topics_scored": total_topics,
            "qrels_queries": total_qrels_queries,
            "topics_equals_qrels_queries": total_topics == total_qrels_queries == 519,
            "missing_from_any_run": sum(c["missing_from_run"] for c in coverage),
            "per_domain": coverage,
        },
        "per_domain_ndcg10": {c["domain"]: c["nDCG@10"] for c in coverage},
        "inputs": inputs,
    }
    out = root / args.out
    Path(str(out) + ".json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    lines = ["# Frozen dev baseline (2026-11-20)", "",
             f"* frozen tag: `{tag}`", f"* old recorded domain-macro nDCG@10: `{args.old_value:.10f}`",
             f"* re-scored now: `{agg['domain_macro']:.10f}` (topic-macro `{agg['topic_macro']:.10f}`)",
             f"* identical: **{baseline['new_reverification']['identical']}**", "",
             f"* coverage: {len(coverage)} domains, {total_topics} queries scored, "
             f"{total_qrels_queries} distinct queries in qrels, missing-from-run "
             f"{baseline['coverage_audit']['missing_from_any_run']}", "",
             "| domain | n | nDCG@10 | topics in run | missing |", "|---|---|---|---|---|"]
    lines += [f"| {c['domain']} | {c['num_topics']} | {c['nDCG@10']:.4f} | "
              f"{c['num_topics_in_run']} | {c['missing_from_run']} |" for c in coverage]
    lines += ["", "## Pinned inputs", ""]
    lines += [f"* `{k}`: {len(v)} file hashes recorded" for k, v in inputs.items() if isinstance(v, dict)]
    lines += ["", f"Full record: `{args.out}.json`", ""]
    Path(str(out) + ".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in baseline.items()
                      if k not in ("inputs", "coverage_audit")}, indent=1)[:2000])
    print("coverage ok:", baseline["coverage_audit"]["topics_equals_qrels_queries"])
    print("wrote", str(out) + ".json")


if __name__ == "__main__":
    main()
