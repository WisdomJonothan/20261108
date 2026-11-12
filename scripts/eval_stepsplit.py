#!/usr/bin/env python
"""Where does the reranking gain come from: single-step or multi-step questions?

Track 1a dev queries carry official step-level evidence (qrels_steps_dev.txt).
This script partitions dev queries by how many distinct evidence steps their
gold answer needs, then reports each system's domain-macro nDCG@10 on every
partition plus the paired, domain-clustered bootstrap CI of the difference.

Usage:
  python scripts/eval_stepsplit.py \
      --a "results/score_sub_zs11_{d}_dev.json" \
      --b "results/score_sub_sft_e2bc_{d}_dev.json" \
      --out results/STEP_SPLIT.md
"""
import argparse
import glob
import io
import json
import os
import random
import sys

ROOT = os.environ.get("RETECO_ROOT", r"D:\RETECO")
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]


def load_per_query(template, domain):
    path = os.path.join(ROOT, template.format(d=domain))
    if not os.path.exists(path):
        return {}
    with io.open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for qid, rec in d.get("per_query", {}).items():
        v = rec.get("nDCG@10")
        if v is not None:
            out[qid] = float(v)
    return out


def load_steps(domain):
    path = os.path.join(ROOT, "data", "track1_tempo", domain, "qrels_steps_dev.txt")
    steps = {}
    if not os.path.exists(path):
        return steps
    with io.open(path, encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 4:
                continue
            qid, sep, step = p[0].rpartition("_step")
            if not sep:
                continue
            steps.setdefault(qid, set()).add(step)
    return steps


def group_of(n_steps):
    if n_steps <= 0:
        return "no-steps"
    if n_steps == 1:
        return "1-step"
    if n_steps == 2:
        return "2-step"
    return "3-step+"


def macro(per_domain_vals):
    ds = [v for v in per_domain_vals if v]
    if not ds:
        return float("nan")
    return sum(ds) / len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline score template with {d}")
    ap.add_argument("--b", required=True, help="candidate score template with {d}")
    ap.add_argument("--out", default="results/STEP_SPLIT.md")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    A, B, STEPS = {}, {}, {}
    for d in DOMAINS:
        A[d] = load_per_query(args.a, d)
        B[d] = load_per_query(args.b, d)
        STEPS[d] = load_steps(d)

    groups = ["1-step", "2-step", "3-step+", "no-steps"]
    lines = []
    lines.append("| partition | queries | domains | A (baseline) | B (candidate) | delta | 95% CI (domain-clustered) |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    rng = random.Random(args.seed)
    summary = {}
    for g in groups:
        pairs = {}  # domain -> list of (a, b)
        for d in DOMAINS:
            lst = []
            for qid, av in A[d].items():
                if qid not in B[d]:
                    continue
                n = len(STEPS[d].get(qid, ()))
                if group_of(n) != g:
                    continue
                lst.append((av, B[d][qid]))
            if lst:
                pairs[d] = lst
        nq = sum(len(v) for v in pairs.values())
        if nq == 0:
            continue
        a_macro = macro([sum(x for x, _ in v) / len(v) for v in pairs.values()])
        b_macro = macro([sum(y for _, y in v) / len(v) for v in pairs.values()])
        delta = b_macro - a_macro

        doms = list(pairs)
        boots = []
        for _ in range(args.iters):
            acc = []
            for d in doms:
                v = pairs[d]
                s = 0.0
                for _i in range(len(v)):
                    x, y = v[rng.randrange(len(v))]
                    s += (y - x)
                acc.append(s / len(v))
            boots.append(sum(acc) / len(acc))
        boots.sort()
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots)) - 1]
        lines.append("| %s | %d | %d | %.4f | %.4f | %+.4f | [%+.4f, %+.4f] |"
                     % (g, nq, len(doms), a_macro, b_macro, delta, lo, hi))
        summary[g] = dict(queries=nq, domains=len(doms), a=round(a_macro, 4),
                          b=round(b_macro, 4), delta=round(delta, 4),
                          ci=[round(lo, 4), round(hi, 4)])

    # share of the overall gain contributed by each partition
    total = 0.0
    for d in DOMAINS:
        qids = [q for q in A[d] if q in B[d]]
        if not qids:
            continue
        total += sum(B[d][q] - A[d][q] for q in qids) / len(qids) / len(DOMAINS)
    lines.append("")
    lines.append("Overall delta (all dev queries, domain-macro): %+.4f" % total)
    lines.append("")
    for g in summary:
        n = summary[g]["queries"]
        if n:
            lines.append("- %s: %d queries (%.1f%%), delta %+.4f"
                         % (g, n, 100.0 * n / max(1, sum(s["queries"] for s in summary.values())),
                            summary[g]["delta"]))

    out = os.path.join(ROOT, args.out)
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote", out)


if __name__ == "__main__":
    sys.exit(main())
