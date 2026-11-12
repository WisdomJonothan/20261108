#!/usr/bin/env python3
"""Summarise the loss/sampler arms (and later the step/query arms) from their logs.

Reads `results/ce_ft/<tag>/train_log.jsonl` for each tag and prints (and
optionally writes) a compact table: per-epoch held-out-train nDCG@10 under both
averages, the epoch the domain-macro rule picks, and the epoch the plain mean
rule would have picked. The two rules differ whenever a large domain and a
small domain disagree, so the table answers "does the selection metric matter?"
without training an extra arm.

Usage:
  python summarize_arms.py --runs lx_possum lx_allpos lx_ndcg lx_sqrt \
      --out results/LOSS_ARMS.md
"""
import argparse
import json
import os

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(tag):
    path = os.path.join(ROOT, "results", "ce_ft", tag, "train_log.jsonl")
    rows = []
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("event") in ("epoch", "epoch0_zeroshot"):
                rows.append(r)
    return rows


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if x != x:
        return "nan"
    return "%.*f" % (nd, x)


def nb(x, nd=4):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return "%.*f" % (nd, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--baseline", default=None,
                    help="tag the delta column is computed against (default: first run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base_tag = args.baseline or args.runs[0]
    base_rows = load(base_tag) or []
    base_map = {r["epoch"]: r.get("val_ndcg10_domain_macro") for r in base_rows}

    lines = ["| arm | ep0 (zero-shot) | e1 | e2 | ep@domain-macro | dm@ep | ep@mean | dm@mean | delta vs " + base_tag + " | train loss (last) |",
             "|" + "---|" * 10]
    payload = {}
    for tag in args.runs:
        rows = load(tag)
        if rows is None:
            lines.append("| %s | (no log) | | | | | | | | |" % tag)
            payload[tag] = None
            continue
        by = {r["epoch"]: r for r in rows}
        if not by:
            lines.append("| %s | (no epoch rows) | | | | | | | | |" % tag)
            payload[tag] = None
            continue

        dm_ep = mean_ep = None
        dms = [e for e in by if e > 0 and isinstance(by[e].get("val_ndcg10_domain_macro"), float)]
        mns = [e for e in by if e > 0 and isinstance(by[e].get("val_ndcg10"), float)]
        if dms:
            dm_ep = max(dms, key=lambda e: by[e]["val_ndcg10_domain_macro"])
        if mns:
            mean_ep = max(mns, key=lambda e: by[e]["val_ndcg10"])
        best = by[dm_ep].get("val_ndcg10_domain_macro") if dm_ep is not None else None
        dm_at_mean = by[mean_ep].get("val_ndcg10_domain_macro") if mean_ep is not None else None
        b0 = base_map.get(dm_ep) if dm_ep in base_map else None
        delta = (best - b0) if isinstance(best, float) and isinstance(b0, float) else None

        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            tag, nb(by.get(0, {}).get("val_ndcg10_domain_macro")),
            nb(by.get(1, {}).get("val_ndcg10_domain_macro")),
            nb(by.get(2, {}).get("val_ndcg10_domain_macro")),
            dm_ep if dm_ep is not None else "-", nb(best),
            mean_ep if mean_ep is not None else "-", nb(dm_at_mean),
            nb(delta), nb(by[max(by)].get("train_loss"))))

        payload[tag] = {
            "best_epoch": dm_ep,
            "best_domain_macro": best,
            "mean_epoch": mean_ep,
            "domain_macro_at_mean_epoch": dm_at_mean,
            "selection_metric_matters": (mean_ep != dm_ep),
            "delta_vs_baseline": delta,
            "per_epoch": {e: {"mean": by[e].get("val_ndcg10"),
                              "domain_macro": by[e].get("val_ndcg10_domain_macro"),
                              "train_loss": by[e].get("train_loss"),
                              "val_auc": by[e].get("val_auc"),
                              "step_hit_rate": by[e].get("step_hit_rate"),
                              "pos_in_top10": by[e].get("pos_in_top10"),
                              "q_with_2pos_top10": by[e].get("q_with_2pos_top10")}
                          for e in by},
            "per_domain_best": (by[dm_ep].get("per_domain") or {}) if dm_ep is not None else {},
        }

    text = "\n".join(lines)
    print(text)
    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        jpath = os.path.splitext(path)[0] + ".json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
        print("\nwritten ->", path, "and", os.path.basename(jpath))


if __name__ == "__main__":
    main()
