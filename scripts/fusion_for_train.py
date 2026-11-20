"""Train-split helper: fuse dense + fold-CE runs with z 1:1.

Restricted to the query ids the fold model was allowed to score, so the fused
run file describes exactly the same query set as the CE run.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm", "iota",
           "law", "monero", "politics", "quant", "travel", "workplace"]


def read_run(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                p = line.split()
            if len(p) < 5:
                continue
            qid, doc, rank, score = p[0], p[2], int(p[3]), float(p[4])
            out.setdefault(qid, []).append((rank, score, doc))
    for q in out:
        out[q].sort(key=lambda x: (x[0], -x[1], x[2]))
    return out


def zscores(rows, depth=100):
    vals = [s for _, s, _ in rows[:depth]]
    if not vals:
        return {}
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    sd = var ** 0.5 or 1.0
    return {d: (s - mu) / sd for _, s, d in rows[:depth]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--domain", required=True)
    ap.add_argument("--dense-tag", default="dense_mxbai")
    ap.add_argument("--ce-tag", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--depth", type=int, default=100)
    args = ap.parse_args()
    R = Path(args.root)
    dense = read_run(R / "runs" / f"{args.dense_tag}_{args.domain}_{args.split}.trec")
    ce = read_run(R / "runs" / f"{args.ce_tag}_{args.domain}_{args.split}_ce.trec")
    assert set(ce) <= set(dense), (args.domain, len(set(ce) - set(dense)))
    out_path = R / "runs" / f"{args.out_tag}_{args.domain}_{args.split}.trec"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q in sorted(ce):
            dz = zscores(dense[q], args.depth)
            cz = zscores(ce[q], args.depth)
            fused = []
            for d, zv in dz.items():
                if d in cz:
                    fused.append((zv + cz[d], d))
            fused.sort(key=lambda x: (-x[0], x[1]))
            for r, (s, d) in enumerate(fused[: args.depth], 1):
                f.write(f"{q}\tQ0\t{d}\t{r}\t{s:.6f}\tfuse_z11\n")
            n += 1
    print(args.domain, "queries", n, "->", out_path)


if __name__ == "__main__":
    main()
