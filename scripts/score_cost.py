#!/usr/bin/env python
"""Assemble the score-vs-cost tables for the frozen replacement study (dev).

Reads every results/replacement_v2/exp_*_dev.json (policy harness, excluding the
probability-dump rerun), exp_budget_cap*.json (budget curve), reranker_*_dev.json
(reranker controls) and the oracle, then appends the decision-rule ablations
(ablations_dev.json) and the gate comparison (gate_eval.json).
"""
import json
from pathlib import Path

R = Path("results/replacement_v2")

FROZEN = 0.35809852326897723


def ci_of(blob):
    if "bootstrap" in blob and isinstance(blob["bootstrap"], dict):
        c = blob["bootstrap"].get("ci95")
        if c:
            return c
    return blob.get("ci95")


def cost_of(blob):
    c = blob.get("cost") or {}
    secs = c.get("seconds") or c.get("seconds_total") or 0.0
    return {
        "prompts": c.get("prompts") or c.get("docs_scored") or c.get("docs"),
        "tokens": c.get("prompt_tokens"),
        "secs": secs,
        "peak_mb": c.get("gpu_peak_mb"),
    }


ROWS = []


def add(name, blob, model, note=""):
    mac = blob.get("macro_policy")
    delta = blob.get("macro_delta", (mac - FROZEN) if mac is not None else None)
    ci = ci_of(blob)
    c = cost_of(blob)
    ROWS.append({
        "name": name, "model": model, "ndcg": mac, "delta": delta, "ci": ci,
        "prompts": c["prompts"], "tokens": c["tokens"], "secs": c["secs"],
        "peak": c["peak_mb"], "note": note,
    })


files = [p for p in sorted(R.glob("exp_*_dev.json")) if "probs" not in p.stem]
for p in files + sorted(R.glob("exp_budget_cap*.json")):
    blob = json.loads(p.read_text())
    for pol, pb in (blob.get("policies") or {}).items():
        name = pol
        note = p.name
        if p.stem.startswith("exp_budget_"):
            name = f"{pol} ({p.stem.replace('exp_budget_', '')})"
        if p.stem == "exp_dense_start_dev":
            name = "direct (dense start)"
            note = "baseline = dense-only 0.2943"
        add(name, pb, blob.get("model", ""), note)

for p in sorted(R.glob("reranker_*_dev.json")):
    blob = json.loads(p.read_text())
    for mode, pb in (blob.get("modes") or {}).items():
        add("qwen3-rr:" + mode, pb, "Qwen3-Reranker-4B", p.name)

oracle = json.loads((R / "oracle_fixed_dev.json").read_text())
ROWS.append({"name": "oracle (cap 3)", "model": "-", "ndcg": oracle["macro_oracle"],
             "delta": oracle["macro_oracle"] - FROZEN, "ci": None,
             "prompts": 0, "tokens": 0, "secs": 0.0, "peak": 0,
             "note": "no judge; upper bound"})

ROWS.sort(key=lambda r: -r["ndcg"])


def fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return f"{v:,}"


lines = []
lines.append("| policy | nDCG@10 | delta | 95% CI | prompts | prompt tokens | s | peak MB | gain / 1k prompts | source |")
lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---|")
for r in ROWS:
    eff = "-"
    if r["prompts"] and r["delta"] is not None:
        eff = f"{r['delta'] / (r['prompts'] / 1000.0):+.4f}"
    ci = "-" if not r["ci"] else f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
    lines.append(
        f"| {r['name']} | {fmt(r['ndcg'])} | {r['delta']:+.4f} | {ci} | "
        f"{fmt(r['prompts'])} | {fmt(r['tokens'])} | {fmt(r['secs'], 1)} | "
        f"{fmt(r['peak'], 0)} | {eff} | {r['note']} |")

ab = R / "ablations_dev.json"
if ab.exists():
    d = json.loads(ab.read_text())
    lines.append("")
    lines.append("## Decision-rule ablations (dev, recomputed from saved per-order probabilities)")
    lines.append("")
    lines.append("| rule | nDCG@10 | delta | 95% CI | help / harm / neutral / no-swap |")
    lines.append("|---|---:|---:|---|---|")
    for rule, v in d["rules"].items():
        sc = v["swap_counts"]
        ci = v["ci95"]
        lines.append(f"| {rule} | {v['macro_policy']:.4f} | {v['macro_delta']:+.4f} | "
                     f"[{ci[0]:+.5f}, {ci[1]:+.5f}] | {sc['helpful']} / {sc['harmful']} / "
                     f"{sc['neutral']} / {sc['no_swap']} |")

g = R / "gate_eval.json"
if g.exists():
    d = json.loads(g.read_text())
    lines.append("")
    lines.append("## Gate comparison (dev; threshold chosen on the 242 calibration queries)")
    lines.append("")
    lines.append("| rule | delta | 95% CI | calibration delta | judge calls |")
    lines.append("|---|---:|---|---:|---:|")
    for rule, v in d.get("dev", {}).items():
        if not isinstance(v, dict) or "macro_delta" not in v:
            continue
        boot = v.get("bootstrap") or {}
        ci = boot.get("ci95")
        ci_s = f"[{ci[0]:+.5f}, {ci[1]:+.5f}]" if ci else "-"
        cal = (d.get("rules", {}).get(rule) or {}).get("calib_macro_delta")
        cal_s = f"{cal:+.5f}" if isinstance(cal, float) else "-"
        stats = v.get("stats") or {}
        calls = stats.get("judge_calls")
        if calls is None and "judge_call_fraction" in v:
            calls = f"{v['judge_call_fraction']:.1%} of pairs"
        lines.append(f"| {rule} | {v['macro_delta']:+.5f} | {ci_s} | {cal_s} | {calls if calls is not None else '-'} |")

md = "\n".join(lines) + "\n"
(R / "SCORE_COST.md").write_text(md)
print(md)
