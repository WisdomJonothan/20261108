"""Paired bootstrap between two training arms on their held-out train queries.

The four-arm comparison (possum / all-positives / nDCG / sqrt-sampler) and the
step-supervision arms are all selected on the *train-internal* split, so the
numbers that choose between them need an uncertainty statement of their own.
`train_ce.py --dump-val-scores` writes one JSON per epoch mapping qid -> nDCG@10;
this script compares two such files with the same paired bootstrap used for dev
(resampling queries, and resampling domains as clusters), and prints the
per-domain differences so a win that comes from one 1-query domain is visible.

    python scripts/compare_val_arms.py --a lx_possum --b lx_allpos --epoch 1 \
        --qids results/val_qids_e2.json
"""
import argparse
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag, epoch):
    p = os.path.join(ROOT, "results", "val_scores_%s_e%d.json" % (tag, epoch))
    if not os.path.exists(p):
        raise SystemExit("missing %s (was the arm trained with --dump-val-scores?)" % p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def qid_domain(qids_path):
    """Map qid -> domain from the pinned split file written by --dump-val-qids."""
    with open(os.path.join(ROOT, qids_path), encoding="utf-8") as f:
        pin = json.load(f)
    m = {}
    for d, qs in pin.get("eval", {}).items():
        for q in qs:
            m[q] = d
    return m


def boot(vals_a, vals_b, dom_of, iters, seed, cluster):
    rng = random.Random(seed)
    keys = list(vals_a)
    if cluster:
        groups = {}
        for q in keys:
            groups.setdefault(dom_of.get(q, "?"), []).append(q)
        gkeys = list(groups)
    deltas = []
    n = len(keys)
    for _ in range(iters):
        if cluster:
            pick = [rng.choice(gkeys) for _ in range(len(gkeys))]
            ks = [q for g in pick for q in groups[g]]
        else:
            ks = [keys[rng.randrange(n)] for _ in range(n)]
        sa = sum(vals_a[q] for q in ks) / len(ks)
        sb = sum(vals_b[q] for q in ks) / len(ks)
        deltas.append(sa - sb)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return sum(deltas) / len(deltas), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline arm tag")
    ap.add_argument("--b", required=True, help="candidate arm tag")
    ap.add_argument("--epoch", type=int, default=-1, help="-1 = last epoch available")
    ap.add_argument("--qids", default="results/val_qids_e2.json")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    def resolve(tag):
        if args.epoch >= 0:
            return args.epoch
        e = 0
        while os.path.exists(os.path.join(ROOT, "results",
                                          "val_scores_%s_e%d.json" % (tag, e))):
            e += 1
        return max(0, e - 1)

    ea, eb = resolve(args.a), resolve(args.b)
    A, B = load(args.a, ea), load(args.b, eb)
    dom_of = qid_domain(args.qids)
    keys = [q for q in A if q in B]
    if not keys:
        raise SystemExit("no overlapping qids between %s and %s" % (args.a, args.b))

    ma = sum(A[q] for q in keys) / len(keys)
    mb = sum(B[q] for q in keys) / len(keys)
    dq = boot(A, B, dom_of, args.iters, args.seed, cluster=False)
    dd = boot(A, B, dom_of, args.iters, args.seed, cluster=True)

    per_dom = {}
    for q in keys:
        d = dom_of.get(q, "?")
        per_dom.setdefault(d, [0.0, 0, 0])
        per_dom[d][0] += A[q] - B[q]
        per_dom[d][1] += 1
        per_dom[d][2] += int(A[q] > B[q])

    print("A=%s (e%d)  B=%s (e%d)  n=%d queries" % (args.a, ea, args.b, eb, len(keys)))
    print("mean A=%.4f  B=%.4f  delta(A-B)=%+.4f" % (ma, mb, ma - mb))
    print("query bootstrap : %+.4f  CI [%+.4f, %+.4f]" % dq)
    print("domain cluster  : %+.4f  CI [%+.4f, %+.4f]" % dd)
    print("per domain (delta, n, wins):")
    for d in sorted(per_dom):
        s, n, w = per_dom[d]
        print("  %-9s %+.4f  n=%2d  wins=%2d" % (d, s / n, n, w))

    if args.json:
        payload = {"a": args.a, "b": args.b, "epoch_a": ea, "epoch_b": eb,
                   "n": len(keys), "mean_a": ma, "mean_b": mb,
                   "delta": ma - mb,
                   "query_bootstrap": {"delta": dq[0], "lo": dq[1], "hi": dq[2]},
                   "domain_bootstrap": {"delta": dd[0], "lo": dd[1], "hi": dd[2]},
                   "per_domain": {d: {"delta": v[0] / v[1], "n": v[1], "wins": v[2]}
                                  for d, v in per_dom.items()}}
        path = os.path.join(ROOT, args.json)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
        print("written ->", path)


if __name__ == "__main__":
    main()
