#!/usr/bin/env python3
"""Produce a submission-ready TREC run for one RETECO Track-1a domain.

Pipeline (the configuration this project settled on):
    mxbai-embed-large-v1 dense top-1000
      -> bge-reranker-v2-m3 on the dense top-100 (title query, window 250)
      -> z-score fusion of dense and CE, weight 1:1 (chosen on TRAIN with the
         official domain-macro statistic; see results/fusion_fit_train_dommacro.json)

Pass --model-ce to score with the supervised checkpoint
(results/ce_ft/v2m3_pool8_e1/best) instead of the zero-shot reranker.

The script deliberately takes a *query file* rather than a split name, so the
same code path is used for dev (sanity check) and for the hidden test queries
once they are released:

    python scripts/make_submission.py --domain bitcoin \
        --queries data/track1_tempo/bitcoin/examples_dev.jsonl \
        --index-data data/track1_tempo/bitcoin \
        --out submissions/bitcoin.trec

Then validate with the official checker:

    python scripts/official/format_checker.py submissions/bitcoin.trec \
        --qrels data/track1_tempo/bitcoin/qrels_dev.txt \
        --corpus data/track1_tempo/bitcoin/documents.jsonl
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.environ.get("RETEco_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib_reteco import load_corpus, read_jsonl  # noqa: E402

RRF_C = 60.0
DENSE_W, CE_W = 1, 2          # train-selected weight (see PHASE8 report)
TOP_DENSE = 1000
TOP_RERANK = 100


def read_trec(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 6:
            out.setdefault(p[0], []).append((int(p[3]), float(p[4]), p[2]))
    for q in out:
        out[q].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--queries", required=True, help="examples_*.jsonl with id+query")
    ap.add_argument("--index-data", required=True, help="domain dir with documents.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=os.path.join(ROOT, "tmp", "submission"))
    ap.add_argument("--model-dense", default=os.path.join(ROOT, "models", "mxbai-embed-large-v1"))
    ap.add_argument("--model-ce", default=os.path.join(ROOT, "models", "bge-reranker-v2-m3"))
    ap.add_argument("--topk", type=int, default=TOP_DENSE)
    ap.add_argument("--fusion", choices=["z", "rrf"], default="z")
    ap.add_argument("--wa", type=float, default=1.0, help="dense weight (z fusion)")
    ap.add_argument("--wb", type=float, default=1.0, help="CE weight (z fusion)")
    ap.add_argument("--tag", default="reteco-mxbai-ce-z")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tag = os.path.basename(os.path.abspath(args.out)).replace(".trec", "")
    dense_run = os.path.join(args.work, f"dense_{args.domain}.trec")
    # the CE cache is keyed by the checkpoint, so a zero-shot cache can never be
    # silently reused for a fine-tuned model (or vice versa)
    base = os.path.basename(os.path.normpath(args.model_ce))
    if base in ("best", "checkpoint", "final"):
        base = os.path.basename(os.path.dirname(os.path.normpath(args.model_ce))) + "_" + base
    ce_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)
    ce_run = os.path.join(args.work, f"ce_{args.domain}_{ce_tag}.trec")

    # 1) dense retrieval over the domain corpus
    if not os.path.isfile(dense_run):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_dense_generic.py"),
               "--index-data", args.index_data, "--split", "dev",
               "--model", args.model_dense, "--tag", "mxbai", "--pooling", "mean",
               "--query-prefix", "Represent this sentence for searching relevant passages: ",
               "--work", args.work, "--out", dense_run, "--topk", str(args.topk),
               "--examples", args.queries,
               "--meta", os.path.join(args.work, f"dense_{args.domain}.meta.json")]
        subprocess.run(cmd, check=True)

    # 2) cross-encoder rerank of the dense top-100
    if not os.path.isfile(ce_run):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_chunk_rerank.py"),
               "--domain", args.domain, "--split", "dev", "--run", dense_run,
               "--k", str(TOP_RERANK), "--model", args.model_ce,
               "--q-mode", "title", "--q-cap", "256", "--win", "250", "--pool", "max",
               "--batch", "32", "--out", ce_run, "--examples", args.queries,
               "--meta", os.path.join(args.work, f"ce_{args.domain}.meta.json")]
        subprocess.run(cmd, check=True)

    # 3) fusion
    dense = read_trec(dense_run)
    ce = read_trec(ce_run)
    fused = {}
    for q in set(dense) | set(ce):
        dlist = [d for (_r, _s, d) in dense.get(q, [])]
        dsc = {d: sc for (_r, sc, d) in dense.get(q, [])}
        csc = {d: sc for (_r, sc, d) in ce.get(q, [])}
        if args.fusion == "rrf":
            rd = {d: i + 1 for i, d in enumerate(dlist)}
            rc = {d: i + 1 for i, d in enumerate(sorted(csc, key=lambda x: -csc[x]))}
            scores = {d: DENSE_W / (RRF_C + rd.get(d, 10 ** 6))
                         + CE_W / (RRF_C + rc.get(d, 10 ** 6))
                      for d in set(rd) | set(rc)}
            fused[q] = sorted(scores.items(), key=lambda t: -t[1])
            continue
        shared = [d for d in dlist if d in csc]
        if not shared:
            fused[q] = [(d, -float(i)) for i, d in enumerate(dlist)]
            continue
        da = np.asarray([dsc[d] for d in shared], dtype=float)
        cb = np.asarray([csc[d] for d in shared], dtype=float)
        da = (da - da.mean()) / da.std() if da.std() > 0 else np.zeros_like(da)
        cb = (cb - cb.mean()) / cb.std() if cb.std() > 0 else np.zeros_like(cb)
        sc = args.wa * da + args.wb * cb
        order = [shared[i] for i in np.argsort(-sc, kind="stable")]
        # documents outside the CE pool keep their dense order, below every scored doc
        tail = [d for d in dlist if d not in csc]
        ranked = [(d, float(v)) for d, v in zip(order, sc[np.argsort(-sc, kind="stable")])]
        ranked += [(d, -1e9 + 1e-6 * i) for i, d in enumerate(tail)]
        fused[q] = ranked

    with open(args.out, "w", encoding="utf-8") as f:
        for q in sorted(fused):
            prev = None
            for rank, (docid, score) in enumerate(fused[q], 1):
                if prev is not None and score >= prev:
                    score = prev - max(1e-9, abs(prev) * 1e-9)
                prev = score
                f.write(f"{q} Q0 {docid} {rank} {score:.6f} {args.tag}" + "\n")

    # 4) coverage sanity check against the query file
    qids = [r["id"] for r in read_jsonl(args.queries)]
    missing = [q for q in qids if q not in fused]
    print(json.dumps({"out": args.out, "topics": len(fused), "missing_topics": missing,
                      "topk": TOP_DENSE, "rerank": TOP_RERANK,
                      "fusion": args.fusion, "weights": [args.wa, args.wb]}, indent=1))
    if missing:
        sys.exit(f"missing topics: {missing}")


if __name__ == "__main__":
    main()
