#!/usr/bin/env python3
"""Compact, human-readable summary of every exp_*.json policy + per-domain table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "results").exists() else Path.home() / "RETECO"
DOMAINS = ["bitcoin","cardano","economics","genealogy","history","hsm","iota",
           "law","monero","politics","quant","travel","workplace"]
V2 = ROOT / "results/replacement_v2"

def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def main():
    frozen = load(ROOT / "results/FROZEN_BASELINE_20261120.json")
    base_dom = frozen["per_domain_ndcg10"] if frozen else {}
    oracle = load(V2 / "oracle_fixed_dev.json")
    out = {"baseline": {"domain_macro": frozen["new_reverification"]["domain_macro"],
                        "topic_macro": frozen["new_reverification"]["topic_macro"],
                        "per_domain": base_dom},
           "oracle": {k: oracle[k] for k in ("macro_baseline","macro_oracle","macro_gain",
                                             "queries_with_beneficial_pair","pairs_positive",
                                             "pairs_negative","n_pairs","n_queries")} if oracle else None,
           "policies": {}, "files": []}
    for f in sorted(V2.glob("exp_*_dev.json")):
        d = load(f)
        if not d or "policies" not in d:
            continue
        out["files"].append(f.name)
        for k, v in d["policies"].items():
            out["policies"][k] = {
                "macro": v["macro_policy"], "delta": v["macro_delta"],
                "ci95": v["bootstrap"]["ci95"], "p_le_0": v["bootstrap"].get("p_le_0"),
                "swap_counts": v["swap_counts"],
                "points_gained": v.get("points_gained", 0.0), "points_lost": v.get("points_lost", 0.0),
                "net_points": v.get("net_points", 0.0),
                "per_domain": v.get("per_domain", {}),
                "cost": v["cost"], "file": f.name,
            }
    (V2 / "summary_dev.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    lines = ["# Dev results — frozen 0.3581 system, one swap per query", "",
             f"baseline domain-macro nDCG@10 = {out['baseline']['domain_macro']:.6f} "
             f"(topic-macro {out['baseline']['topic_macro']:.6f}); 519 queries / 13 domains", ""]
    if out["oracle"]:
        o = out["oracle"]
        lines += [f"oracle ceiling of the frozen pool = {o['macro_oracle']:.6f} "
                  f"({o['macro_gain']:+.6f}), {o['queries_with_beneficial_pair']}/{o['n_queries']} queries "
                  f"have a beneficial pair, {o['pairs_positive']} vs {o['pairs_negative']} pairs", ""]
    cols = [("direct","direct"),("contextual","context"),("perdoc","per-doc"),("active","active")]
    have = [(k,l) for k,l in cols if k in out["policies"]]
    lines += ["| arm | nDCG@10 | Δ | 95% CI | beneficial | harmful | neutral | no swap | gained | lost | prompts | tokens | s | peak MB |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, label in have:
        p = out["policies"][k]; c = p["cost"]
        n_prompts = c.get("prompts", c.get("docs", 0)); tok = c.get("prompt_tokens", 0)
        secs = c.get("seconds_total", c.get("seconds", 0.0)); peak = c.get("gpu_peak_mb", 0.0)
        sc = p["swap_counts"]
        lines.append(f"| {label} | {p['macro']:.4f} | {p['delta']:+.4f} | "
                     f"[{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] | {sc.get('helpful',0)} | {sc.get('harmful',0)} | "
                     f"{sc.get('neutral',0)} | {sc.get('no_swap',0)} | {p['points_gained']:+.3f} | "
                     f"{p['points_lost']:+.3f} | {n_prompts:,} | {tok:,} | {secs:.0f} | {peak:.0f} |")
    lines += ["", "## Per-domain nDCG@10", "",
              "| domain | baseline | " + " | ".join(l for _, l in have) + " |",
              "|---|---:|" + "---:|" * len(have)]
    for d in DOMAINS:
        row = [d, f"{base_dom.get(d,0):.4f}"]
        for k, _ in have:
            row.append(f"{out['policies'][k]['per_domain'].get(d, {}).get('policy', float('nan')):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    (V2 / "SUMMARY_DEV.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
