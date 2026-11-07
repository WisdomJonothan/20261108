"""Paired comparison of two retrievers' runs (per-query nDCG@10).

Usage:
  python compare_retrievers.py --base score_dense_bge_bitcoin_dev.json \
      --alt score_dense_bgelarge_bitcoin_dev.json --json out.json
"""
import argparse
import json
import random
import statistics


def load(p):
    return json.load(open(p, encoding="utf-8"))["per_query"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()
    random.seed(0)
    b, a = load(args.base), load(args.alt)
    qs = [q for q in b if q in a]
    diffs = [a[q]["nDCG@10"] - b[q]["nDCG@10"] for q in qs]
    n = len(diffs)
    m = statistics.mean(diffs)
    sd = statistics.pstdev(diffs)
    boots = sorted(sum(diffs[random.randrange(n)] for _ in range(n)) / n
                   for _ in range(args.boot))
    out = {
        "n": n,
        "base_mean": statistics.mean(b[q]["nDCG@10"] for q in qs),
        "alt_mean": statistics.mean(a[q]["nDCG@10"] for q in qs),
        "delta": m,
        "t": m / (sd / n ** 0.5) if sd > 0 else None,
        "ci": [boots[int(0.025 * args.boot)], boots[int(0.975 * args.boot)]],
        "p_gt0": sum(1 for x in boots if x > 0) / len(boots),
        "better": sum(1 for d in diffs if d > 1e-9),
        "worse": sum(1 for d in diffs if d < -1e-9),
        "tie": sum(1 for d in diffs if abs(d) <= 1e-9),
    }
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
