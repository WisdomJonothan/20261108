"""Compare the evidence-window arms at an equal 112-word display budget.

Arms:
  direct    the frozen harness (window chosen by content-term overlap)
  cond      same window size, window chosen by the question's conditions
            (content terms + numbers/dates + capitalised entities)
  two       two 56-word windows: the best condition window plus the best
            non-adjacent one
  wide400   the 400-word control (1.55x the prompt tokens), for reference only

Writes results/replacement_v2/win_table.md and win_table.json.
"""
import json
from pathlib import Path

R = Path(r"D:/RETECO/github_repo/results/replacement_v2")
BASE = 0.35809852326897723


def load_direct():
    d = json.loads((R / "exp_direct_context_dev.json").read_text(encoding="utf-8"))
    p = d["policies"]["direct"]
    return {
        "nDCG@10": p["macro_policy"], "delta": p["macro_delta"],
        "ci": p["bootstrap"]["ci95"], "swaps": p["swap_counts"],
        "gained": p["points_gained"], "lost": p["points_lost"],
        "prompts": p["cost"]["prompts"], "tokens": p["cost"]["prompt_tokens"],
        "seconds": p["cost"]["seconds_total"], "per_query": None,
    }


def load_alt(path):
    d = json.loads((R / path).read_text(encoding="utf-8"))
    if "policies" in d:
        p = d["policies"]["direct"]
        return {"nDCG@10": p["macro_policy"], "delta": p["macro_delta"],
                "ci": p["bootstrap"]["ci95"], "swaps": p["swap_counts"],
                "gained": p["points_gained"], "lost": p["points_lost"],
                "prompts": p["cost"].get("prompts"), "tokens": p["cost"].get("prompt_tokens"),
                "seconds": p["cost"].get("seconds_total"), "per_query": None}
    return {"nDCG@10": d["macro_policy"], "delta": d["macro_delta"], "ci": d["bootstrap"]["ci95"],
            "swaps": d["swap_counts"], "gained": d["points_gained"], "lost": d["points_lost"],
            "prompts": d["cost"].get("prompts"), "tokens": d["cost"].get("prompt_tokens"),
            "seconds": d["cost"].get("seconds_total"), "per_query": d.get("per_query")}


def per_query_delta(path):
    d = json.loads((R / path).read_text(encoding="utf-8"))
    if "policies" in d:
        out = {}
        for q in d["policies"]["direct"].get("queries", []):
            out[(q["domain"], str(q["qid"]))] = q["delta"]
        if out:
            return out
        # fall back to pair records (one chosen swap per query at most)
        acc = {}
        for rec in d["policies"]["direct"]["pair_records"]:
            k = (rec["domain"], str(rec["qid"]))
            acc.setdefault(k, 0.0)
            if rec.get("chosen"):
                acc[k] += rec["delta_ndcg"]
        return acc
    return {(q["domain"], str(q["qid"])): q["delta"] for q in d["per_query"]}


def main():
    import numpy as np

    arms = {"direct": load_direct(),
            "cond": load_alt("exp_cond_win_dev.json"),
            "two": load_alt("exp_two_win_dev.json"),
            "wide400": load_alt("exp_wide_win400_dev.json")}
    rows = []
    for name, a in arms.items():
        sc = a["swaps"]
        rows.append((name, a["nDCG@10"], a["delta"], a["ci"][0], a["ci"][1],
                     sc.get("no_swap"), sc.get("helpful"), sc.get("harmful"),
                     a["gained"], a["lost"], a["prompts"], a["tokens"], a["seconds"]))
    lines = ["| arm | nDCG@10 | delta | 95% CI | no-swap | helpful | harmful | gained | lost | prompts | tokens | s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append("| %s | %.4f | %+.4f | [%+.4f, %+.4f] | %d | %d | %d | %+.2f | %+.2f | %s | %s | %.0f |" % (
            r[0], r[1], r[2], r[3], r[4], r[5] or 0, r[6] or 0, r[7] or 0,
            r[8], r[9], r[10], r[11], r[12]))

    # paired intervals against direct, resampling queries within domains
    base = per_query_delta("exp_direct_context_dev.json")
    rng = np.random.default_rng(20261120)
    dom = {}
    for (d, q) in base:
        dom.setdefault(d, []).append((d, q))
    paired = {}
    for name, path in (("cond", "exp_cond_win_dev.json"), ("two", "exp_two_win_dev.json")):
        alt = per_query_delta(path)
        keys = [k for k in base if k in alt]
        diffs = {k: alt[k] - base[k] for k in keys}
        draws = []
        for _ in range(10000):
            pick = []
            for d, ks in dom.items():
                ks = [k for k in ks if k in diffs]
                if not ks:
                    continue
                idx = rng.integers(0, len(ks), size=len(ks))
                pick.extend(ks[i] for i in idx)
            draws.append(float(np.mean([diffs[k] for k in pick])))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        better = sum(1 for k in keys if diffs[k] > 1e-9)
        worse = sum(1 for k in keys if diffs[k] < -1e-9)
        paired[name] = {"n": len(keys), "delta": float(np.mean([diffs[k] for k in keys])),
                        "ci95": [float(lo), float(hi)], "better": better, "worse": worse}
        lines.append("")
        lines.append("paired %s - direct: %+.6f, CI [%+.6f, %+.6f], better %d / worse %d of %d"
                     % (name, paired[name]["delta"], lo, hi, better, worse, len(keys)))

    md = "## Evidence-window comparison (112-word display budget)\n\n" + "\n".join(lines) + "\n"
    (R / "win_table.md").write_text(md, encoding="utf-8")
    (R / "win_table.json").write_text(json.dumps(
        {"arms": {k: {kk: vv for kk, vv in v.items() if kk != "per_query"} for k, v in arms.items()},
         "paired_vs_direct": paired, "baseline": BASE}, indent=1), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
