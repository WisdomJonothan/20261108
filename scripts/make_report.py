#!/usr/bin/env python3
"""Assemble results/replacement_v2/REPORT.md from the frozen study artefacts."""
import argparse
import json
from pathlib import Path

DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]
FMT = "{:+.4f}"


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt_ci(ci):
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.root
    v2 = root / "results/replacement_v2"
    frozen = load(root / "results/FROZEN_BASELINE_20261120.json")
    cand = load(v2 / "candidates_dev.json")
    oracle = load(v2 / "oracle_fixed_dev.json")
    exp = {"policies": {}, "n_queries": 519}
    for f in sorted(v2.glob("exp_*_dev.json")):
        d = load(f)
        if not d:
            continue
        exp["n_queries"] = d.get("n_queries", exp["n_queries"])
        exp["policies"].update(d.get("policies", {}))
        exp.setdefault("files", []).append(f.name)
    rer = load(v2 / "reranker_controls_dev.json")
    lines = ["# Bounded replacement study — results (dev, 519 queries)", ""]

    lines += ["## 0. Frozen baseline", ""]
    if frozen:
        nr = frozen["new_reverification"]
        lines += [f"* re-scored domain-macro nDCG@10 = `{nr['domain_macro']:.10f}` "
                  f"(recorded `{frozen['old_recorded']['domain_macro']:.10f}`), identical = "
                  f"**{nr['identical']}**",
                  f"* coverage: {frozen['coverage_audit']['domains']} domains, "
                  f"{frozen['coverage_audit']['topics_scored']} queries, missing from runs = "
                  f"{frozen['coverage_audit']['missing_from_any_run']}",
                  f"* hashes: `results/FROZEN_BASELINE_20261120.json`"]
    lines += ["", "## 1. Frozen candidate protocol and its ceiling", ""]
    if cand and oracle:
        pr = cand["protocol"]
        lines += [f"* {pr['candidate_rule']}", f"* {pr['comparison_rule']}", f"* {pr['swap_rule']}",
                  f"* queries {cand['n_queries']}, pairs {cand['n_pairs']}, "
                  f"queries with 3 candidates {cand['queries_with_3_candidates']}",
                  "", "| | value |", "|---|---:|",
                  f"| baseline | {oracle['macro_baseline']:.4f} |",
                  f"| oracle (<=1 swap, frozen pairs) | {oracle['macro_oracle']:.4f} |",
                  f"| ceiling gain | {oracle['macro_gain']:+.4f} |",
                  f"| queries with a beneficial pair | {oracle['queries_with_beneficial_pair']} / {oracle['n_queries']} |",
                  f"| beneficial / harmful pairs | {oracle['pairs_positive']} / {oracle['pairs_negative']} of {oracle['n_pairs']} |",
                  "", "> The earlier 0.4098 figure came from an *uncapped* candidate pool and is not "
                  "the ceiling of this protocol."]
    lines += ["", "## 2. Direct vs Context (Qwen3-4B-Instruct-2507, threshold 0, one swap/query)", ""]
    if exp["policies"]:
        lines += ["| arm | macro nDCG@10 | delta | 95% CI | helpful | harmful | neutral | no swap | prompts | prompt tok | seconds | peak MB |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm, p in exp["policies"].items():
            c = p["cost"]
            n_prompts = c.get("prompts", c.get("docs", 0))
            n_tok = c.get("prompt_tokens", 0)
            secs = c.get("seconds_total", c.get("seconds", 0.0))
            peak = c.get("gpu_peak_mb", 0.0)
            lines.append(f"| {arm} | {p['macro_policy']:.4f} | {p['macro_delta']:+.4f} | "
                         f"{fmt_ci(p['bootstrap']['ci95'])} | {p['swap_counts'].get('helpful', 0)} | "
                         f"{p['swap_counts'].get('harmful', 0)} | {p['swap_counts'].get('neutral', 0)} | "
                         f"{p['swap_counts'].get('no_swap', 0)} | {n_prompts:,} | {n_tok:,} | "
                         f"{secs:.0f} | {peak:.0f} |")
        lines += ["", "| arm | points gained | points lost | net | pairs agreeing | pairs eligible |",
                  "|---|---:|---:|---:|---:|---:|"]
        for arm, p in exp["policies"].items():
            lines.append(f"| {arm} | {p['points_gained']:+.3f} | {p['points_lost']:+.3f} | "
                         f"{p['net_points']:+.3f} | {p['agreement_pairs']} / {exp['n_queries'] * 9} | "
                         f"{p['eligible_pairs']} |")
        lines += ["", "### Per-domain nDCG@10", "",
                  "| domain | n | baseline | " + " | ".join(exp["policies"]) + " |",
                  "|---|---:|---:|" + "---:|" * len(exp["policies"])]
        for d in DOMAINS:
            row = [f"| {d} |"]
            first = next(iter(exp["policies"].values()))
            if d not in first["per_domain"]:
                continue
            cells = [str(first["per_domain"][d]["n"]), f"{first['per_domain'][d]['baseline']:.4f}"]
            cells += [f"{exp['policies'][a]['per_domain'].get(d, {}).get('policy', float('nan')):.4f}" for a in exp["policies"]]
            lines.append(row[0] + " | ".join(cells) + " |")
    lines += ["", "## 3. Reranker controls (score vs cost)", ""]
    if rer:
        lines += ["| mode | macro nDCG@10 | delta | 95% CI | helpful | harmful | neutral | no swap | docs scored | seconds | docs/s | peak MB |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for mode, p in rer["modes"].items():
            c = p["cost"]
            lines.append(f"| {mode} | {p['macro_policy']:.4f} | {p['macro_delta']:+.4f} | "
                         f"{fmt_ci(p['ci95'])} | {p['swap_counts'].get('helpful', 0)} | "
                         f"{p['swap_counts'].get('harmful', 0)} | {p['swap_counts'].get('neutral', 0)} | "
                         f"{p['swap_counts'].get('no_swap', 0)} | {c['docs_scored']:,} | "
                         f"{c['seconds']:.0f} | {c['docs_per_s']:.0f} | {c['gpu_peak_mb']:.0f} |")
    out = v2 / "REPORT.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
