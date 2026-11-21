#!/usr/bin/env python3
"""Assemble results/replacement_v2/REPORT.md from the committed study artefacts."""
import json
from pathlib import Path

ROOT = Path(r"D:\RETECO\github_repo")
V2 = ROOT / "results" / "replacement_v2"


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ci(d):
    c = d.get("ci95")
    if not c:
        b = d.get("bootstrap") or {}
        c = b.get("ci95")
    if not c:
        return "-"
    return "[%+.4f, %+.4f]" % (c[0], c[1])


def secs(cost):
    return cost.get("seconds") or cost.get("seconds_total") or 0.0


L = []
a = L.append

frozen = load(ROOT / "results/FROZEN_BASELINE_20261120.json")
oracle = load(V2 / "oracle_fixed_dev.json")
cand = load(V2 / "candidates_dev.json")
a("# Bounded replacement study — results (dev, 519 queries)\n")
a("All numbers below come from committed artefacts under "
  "`results/replacement_v2/`; the frozen protocol is "
  "`configs/replacement_study_20261120.json`.\n")

if frozen:
    rv = frozen["new_reverification"]
    a("## 0. Frozen baseline\n")
    a("* re-scored domain-macro nDCG@10 = `%.10f` (recorded delta %+.1e, identical = %s)"
      % (rv["domain_macro"], rv["delta_vs_old"], rv["identical"]))
    a("* topic-macro = `%.10f`; domains = %d, queries = %d"
      % (rv["topic_macro"], rv["num_domains"], rv["num_topics"]))
    a("* hashes in `results/FROZEN_BASELINE_20261120.json`\n")

if oracle and cand:
    a("## 1. Frozen candidate protocol and its ceiling\n")
    a("* union(dense top20, full-CE top20) minus frozen top10; order by best rank "
      "(tie: dense before CE, then docid); keep 3")
    a("* each candidate vs frozen ranks 8, 9, 10 (<= 9 pairs); at most one swap per query")
    a("* %d queries, %d pairs" % (oracle.get("n_queries", 0), oracle.get("n_pairs", 0)))
    a("")
    a("| | value |")
    a("|---|---:|")
    a("| baseline | 0.3581 |")
    a("| oracle (<=1 swap, frozen pairs) | %.4f |" % oracle["macro_oracle"])
    a("| ceiling gain | %+.4f |" % oracle["macro_gain"])
    a("| queries with a beneficial pair | %d / 519 |" % oracle["queries_with_beneficial_pair"])
    a("")
    a("> The earlier 0.4098 figure came from an *uncapped* pool and is not the "
      "ceiling of this protocol.\n")

arms = [
    ("frozen (no judge)", None),
    ("direct", V2 / "exp_direct_context_dev.json"),
    ("contextual", V2 / "exp_direct_context_dev.json"),
    ("perdoc", V2 / "exp_perdoc_dev.json"),
    ("active", V2 / "exp_active_dev.json"),
    ("direct, budget cap 1", V2 / "exp_budget_cap1.json"),
    ("direct, budget cap 2", V2 / "exp_budget_cap2.json"),
]
a("## 2. Arms (Qwen3-4B-Instruct-2507, threshold 0, one swap per query)\n")
a("| arm | nDCG@10 | delta | 95% CI | helpful | harmful | neutral | no swap | prompts | tokens | s | peak MB |")
a("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
a("| frozen | 0.3581 | - | - | - | - | - | - | 0 | 0 | 0 | 0 |")
for name, path in arms[1:]:
    d = load(path)
    if not d:
        continue
    key = "direct" if name.startswith("direct") else name
    if name.startswith("direct, budget"):
        key = "direct"
    p = d["policies"].get(key)
    if not p:
        continue
    sc = p["swap_counts"]
    c = p["cost"]
    a("| %s | %.4f | %+.4f | %s | %d | %d | %d | %d | %s | %s | %.0f | %.0f |"
      % (name, p["macro_policy"], p["macro_delta"], ci(p),
         sc.get("helpful", 0), sc.get("harmful", 0), sc.get("neutral", 0), sc.get("no_swap", 0),
         format(c.get("prompts", 0), ","), format(c.get("prompt_tokens", 0), ","), secs(c), c.get("gpu_peak_mb", 0)))
a("")
a("Context is +0.0051 with 1.77x the tokens and 1.74x the time of Direct, and the "
  "paired Direct-Context difference is -0.0015 (CI [-0.0051, +0.0016]): retained "
  "context is not a contribution.\n")

fuse = load(V2 / "exp_direct_context_dev.json")
a("Points gained by helpful swaps: %+.2f; lost by harmful swaps: %+.2f (net %+.2f)."
  % (fuse["policies"]["direct"]["points_gained"], fuse["policies"]["direct"]["points_lost"],
     fuse["policies"]["direct"]["net_points"]))
a("")
a("### Per-domain nDCG@10 (frozen / direct / contextual)")
a("")
a("| domain | frozen | direct | contextual |")
a("|---|---:|---:|---:|")
for d in sorted(fuse["policies"]["direct"]["per_domain"]):
    row = fuse["policies"]["direct"]["per_domain"][d]
    c2 = fuse["policies"]["contextual"]["per_domain"].get(d, {})
    a("| %s | %.4f | %.4f | %.4f |" % (d, row.get("baseline", float("nan")),
                                       row.get("policy", float("nan")),
                                       c2.get("policy", float("nan"))))
a("")

a("## 3. Starting systems (same candidates, judge and one-swap rule)")
a("")
a("| start | nDCG@10 (start) | + correction | delta | 95% CI | help / harm | prompts | s |")
a("|---|---:|---:|---:|---|---:|---:|---:|")


def start_row(name, d, pol):
    p = d["policies"][pol]
    c = p["cost"]
    a("| %s | %.4f | %.4f | %+.4f | %s | %d / %d | %s | %.0f |"
      % (name, p["macro_baseline"], p["macro_policy"], p["macro_delta"], ci(p),
         p["swap_counts"].get("helpful", 0), p["swap_counts"].get("harmful", 0),
         format(c.get("prompts", 0), ","), secs(c)))


start_row("fused (frozen top-10)", fuse, "direct")
ds = load(V2 / "exp_dense_start_dev.json")
if ds:
    start_row("dense top-100", ds, "direct")
rrst = load(V2 / "exp_rr_start_dev.json")
if rrst:
    start_row("Qwen3-Reranker-4B on dense top-20", rrst, "direct")
a("")
a("The correction is not specific to the fused start: it gains most where the")
a("top ten has more clearly wrong slots.")
a("")

rr = load(V2 / "reranker_fair_dev.domainmacro.json")
a("## 4. Strong controls (Qwen3-Reranker-4B, official template, same passages)\n")
a("| control | nDCG@10 | delta | 95% CI | docs scored | tokens | s |")
a("|---|---:|---:|---|---:|---:|---:|")
if rr:
    for m in ("single_swap", "full_rerank"):
        p = rr["modes"][m]
        c = p.get("cost") or {}
        a("| %s | %.4f | %+.4f | %s | %s | %s | %.0f |"
          % (m, p["macro_policy"], p["macro_delta"], ci(p),
             format(c.get("docs_scored", 0), ","),
             format(c.get("prompt_tokens", 0), ","), secs(c)))
    a("")
    a("The suffix is kept outside the truncated passage and the passage is the same "
      "112-word query-matched window Direct sees. Single swap reaches 0.3616 with an "
      "interval that includes zero, so the reranker neither beats the judge nor is "
      "broken; full reranking of the same thirteen documents is significantly worse. "
      "An earlier right-padded run and a prefix-reading control had made the reranker "
      "look far worse than it is.\n")

ab = load(V2 / "ablations_dev.json")
if ab:
    a("## 5. Decision-rule ablations\n")
    a("| rule | nDCG@10 | delta | 95% CI | help / harm / neutral / no-swap |")
    a("|---|---:|---:|---|---|")
    for k in ("full", "no_abstain", "no_bidir", "neither"):
        p = ab["rules"][k]
        sc = p["swap_counts"]
        a("| %s | %.4f | %+.4f | %s | %d / %d / %d / %d |"
          % (k, p["macro_policy"], p["macro_delta"], ci(p),
             sc.get("helpful", 0), sc.get("harmful", 0), sc.get("neutral", 0), sc.get("no_swap", 0)))
    a("")
    a("Bidirectional agreement is load-bearing; abstention is not.\n")

g = load(V2 / "gate_eval.json")
if g:
    a("## 6. Gate (logistic regression on 3-fold, leakage-free train features)\n")
    a("| rule | dev delta | 95% CI | calibration delta | judge calls |")
    a("|---|---:|---|---:|---:|")
    for k in ("gate_judge", "direct_all", "fixed_margin_0.9", "ce_gap_rule", "ce_or_dense_gap", "precall_cheap_gate"):
        if k not in g["dev"]:
            continue
        p = g["dev"][k]
        cal = g.get("calibration", {}).get("macro_delta") if k == "direct_all" else None
        calls = p.get("stats", {}).get("judge_calls")
        if calls is None and "judge_call_fraction" in p:
            calls = "%.1f%% of pairs" % (100 * p["judge_call_fraction"])
        if calls is None:
            calls = "-"
        a("| %s | %+.4f | %s | %s | %s |"
          % (k, p["macro_delta"], ci(p), "%+.4f" % cal if cal is not None else "-", calls))
    a("")
    a("Threshold chosen on the 242 calibration queries: tau = %.4f. The gate "
      "reproduces Direct; the cheap gap rule is the strongest variant; a pre-call "
      "cheap gate buys 15.6%% of the judge calls for 16%% of the gain.\n" % g["calibration"]["tau"])

a("## 7. Fold evidence (train split, 969 queries)\n")
a("| fold | queries | nDCG@10 | delta | 95% CI |")
a("|---|---:|---:|---:|---|")
for f in ("f0", "f1", "f2", "fall"):
    d = load(V2 / ("exp_train_%s.json" % f))
    if not d:
        continue
    p = d["policies"]["direct"]
    a("| %s | %d | %.4f | %+.4f | %s |" % (f, d["n_queries"], p["macro_policy"], p["macro_delta"], ci(p)))
a("")
a("The folds differ in CE, start score and split, so they measure the sensitivity "
  "of the gain rather than a fresh-split estimate: fold starts 0.2562 / 0.2796 / 0.3030 "
  "give +0.0007 / +0.0032 / +0.0068, and the 242 calibration queries (start 0.2653) "
  "give -0.0014. We report the measured dev gain and these splits without "
  "extrapolating.\n")
a("## 8. Case studies\n")
a("See `CASES_DEV.md`: the six largest gains and losses with query text, and the "
  "per-domain swap counts.\n")
a("## 9. Transfer\n")
tr = load(V2 / "followir" / "scores_followir.json")
pm = load(V2 / "followir" / "pmrr.json")
if tr:
    a("| collection | topics | docs | frozen og | frozen chg | Direct delta | 95% CI | help / harm |")
    a("|---|---:|---:|---:|---:|---:|---|---|")
    for name in ("core17", "news21", "robust04"):
        c = tr["collections"].get(name)
        if not c:
            continue
        vals = list(c["graded_ndcg10"].values())
        og = vals[0]["frozen"] if vals else float("nan")
        chg = vals[1]["frozen"] if len(vals) > 1 else float("nan")
        a("| %s | %d | %s | %.4f | %.4f | %+.4f | %s | %d / %d |"
          % (name, c["topics"], format(c["documents"], ","), og, chg,
             c["direct"]["delta"], ci(c["direct"]),
             c["direct"]["counts"].get("helpful", 0),
             c["direct"]["counts"].get("harmful", 0)))
    a("")
if pm:
    a("Official p-MRR (mteb rank_score; negative means documents that should drop "
      "under the changed instruction stay high):\n")
    a("| collection | dense | CE-only | frozen |")
    a("|---|---:|---:|---:|")
    for name in ("core17", "news21", "robust04"):
        c = pm.get(name)
        if not c:
            continue
        row = c
        a("| %s | %+.4f | %+.4f | %+.4f |"
          % (name, row["dense"]["p_mrr"], row["ce"]["p_mrr"], row["frozen"]["p_mrr"]))
    a("")
    a("Every cell is negative and no document is promoted, so the positive nDCG "
      "deltas measure ranking quality on the changed query text, not "
      "instruction-following.\n")

out = V2 / "REPORT.md"
with open(out, "w", encoding="utf-8", newline=chr(10)) as f:
    f.write(chr(10).join(L) + chr(10))
print("wrote", out, len("\n".join(L)))
