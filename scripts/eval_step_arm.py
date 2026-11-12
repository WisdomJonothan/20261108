"""One-command evaluation of a step-supervision arm against its baseline.

Reads the artefacts produced by ``ops/final_eval.sh`` (aggregate JSON, paired
bootstrap JSON against the frozen arm, multi-evidence JSON per arm) and the
step-partition split from ``scripts/eval_stepsplit.py``, then prints the
pre-registered verdict of ``docs/RESEARCH_PLAN_NEXT.md`` section 4b:

  A  main criterion      dev domain-macro gain over the baseline >= +0.005 and
                         the domain-clustered 95% CI excludes 0
  A  mechanism criterion step_cov@10 and pos_in_top10 must rise, and
                         multi_at_10 must rise by at least 2pp
  A  negative case       domain-macro flat but only multi_at_10 up  -> unresolved

Usage:
  python scripts/eval_step_arm.py --base sft_e2bc --cand sx_l03
  python scripts/eval_step_arm.py --base sft_e2bc --cand sx_l10 --out results/STEP_ARM_sx_l10.md
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pairs(ci):
    if not ci:
        return None
    lo = ci.get("ci_lo", ci.get("ci_low", ci.get("lo")))
    hi = ci.get("ci_hi", ci.get("ci_high", ci.get("hi")))
    return (lo, hi) if lo is not None and hi is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sft_e2bc")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stepsplit-iters", type=int, default=2000)
    args = ap.parse_args()

    agg_b = load(os.path.join(ROOT, "results", "aggregate_sub_%s_dev.json" % args.base))
    agg_c = load(os.path.join(ROOT, "results", "aggregate_sub_%s_dev.json" % args.cand))
    if not agg_c:
        raise SystemExit("missing results/aggregate_sub_%s_dev.json" % args.cand)

    boot = (load(os.path.join(ROOT, "results", "boot_dom_%s_vs_frozen.json" % args.cand))
            or load(os.path.join(ROOT, "results", "boot_dom_%s_vs_zs.json" % args.cand)))
    me_b = load(os.path.join(ROOT, "results", "MULTIEVIDENCE_%s.json" % args.base))
    me_c = load(os.path.join(ROOT, "results", "MULTIEVIDENCE_%s.json" % args.cand))

    base_dm = (agg_b or {}).get("domain_macro")
    cand_dm = agg_c.get("domain_macro")
    delta = None if base_dm is None else cand_dm - base_dm

    lines = []
    w = lines.append
    w("## %s vs %s" % (args.cand, args.base))
    w("")
    w("| | dev domain-macro nDCG@10 |")
    w("|---|---:|")
    w("| %s (baseline) | %s |" % (args.base, "%.4f" % base_dm if base_dm else "n/a"))
    w("| %s | %.4f |" % (args.cand, cand_dm))
    if delta is not None:
        w("| delta | %+.4f |" % delta)
    w("")

    verdict_main = "unknown"
    if boot:
        q = pairs(boot.get("query_bootstrap"))
        d = pairs(boot.get("domain_bootstrap"))
        w("paired bootstrap (`%s`): delta %+.4f, query CI %s, domain CI %s, "
          "%d better / %d worse" % (
              os.path.basename(str(boot.get("label", "bootstrap"))),
              boot.get("delta", float("nan")),
              "[%+.4f,%+.4f]" % q if q else "n/a",
              "[%+.4f,%+.4f]" % d if d else "n/a",
              boot.get("better", 0), boot.get("worse", 0)))
        w("")
        if d and d[0] is not None:
            verdict_main = ("confirmed" if (boot["delta"] >= 0.005 and d[0] > 0)
                            else ("flat" if d[0] <= 0 <= d[1] else "negative"))
    else:
        w("paired bootstrap: missing (run `ops/final_eval.sh`)")
        w("")

    mech_rows = []
    if me_b and me_c:
        a, b = me_b["domain_macro"], me_c["domain_macro"]
        for key, fmt in (("step_cov", "%.4f"), ("step_recall", "%.4f"),
                         ("gold_mean", "%.4f"), ("multi_at_k", "%.4f")):
            mech_rows.append((key, a.get(key), b.get(key)))
    if mech_rows:
        w("| mechanism | %s | %s | delta |" % (args.base, args.cand))
        w("|---|---:|---:|---:|")
        for k, x, y in mech_rows:
            if x is None or y is None:
                w("| %s | n/a | n/a | |" % k)
            else:
                suffix = "pp" if k == "multi_at_k" else ""
                scale = 100.0 if k == "multi_at_k" else 1.0
                w("| %s | %s | %s | %+.4f%s |" % (k, fmt % x, fmt % y,
                                                  (y - x) * scale, suffix))
        w("")
        verdict_mech = ("up" if (me_c["domain_macro"].get("step_cov") or 0) > (me_b["domain_macro"].get("step_cov") or 0)
                        and (me_c["domain_macro"].get("multi_at_k") or 0) >= (me_b["domain_macro"].get("multi_at_k") or 0) + 0.02
                        else "not moved")
    else:
        w("multi-evidence JSON missing for one arm (`scripts/dev_multievidence.py`)")
        w("")
        verdict_mech = "unknown"

    split_out = os.path.join(ROOT, "results", "STEP_SPLIT_%s_vs_%s.md" % (args.cand, args.base))
    try:
        subprocess.run([PY, os.path.join(ROOT, "scripts", "eval_stepsplit.py"),
                        "--a", "results/score_sub_%s_{d}_dev.json" % args.base,
                        "--b", "results/score_sub_%s_{d}_dev.json" % args.cand,
                        "--iters", str(args.stepsplit_iters),
                        "--out", split_out],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        with open(split_out, encoding="utf-8") as f:
            table = [l for l in f.read().splitlines() if l.startswith("|")]
        w("### step partition (two-step rows are the pre-registered check)")
        w("")
        lines.extend(table)
        w("")
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        w("step split failed: %s" % exc)
        w("")

    w("### verdict against RESEARCH_PLAN_NEXT 4b")
    w("")
    w("* main criterion (`+0.005`, CI excluding 0): **%s**" % verdict_main)
    w("* mechanism criterion (step_cov and multi_at_10 move): **%s**" % verdict_mech)
    if verdict_main != "confirmed":
        w("* per the pre-registration a flat or negative main criterion is reported as "
          "*not established*, whatever the mechanism table shows")
    w("")

    text = "\n".join(lines)
    out = args.out or os.path.join(ROOT, "results", "STEP_ARM_%s.md" % args.cand)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
