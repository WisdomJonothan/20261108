#!/usr/bin/env python3
"""Produce a submission-ready TREC run for one RETECO Track-1a domain.

Pipeline (the configuration this project settled on):
    mxbai-embed-large-v1 dense top-1000
      -> bge-reranker-v2-m3 on the dense top-100 (title query, window 250)
      -> z-score fusion of dense and CE, weight 1:1 (chosen on TRAIN with the
         official domain-macro statistic; see results/fusion_fit_train_dommacro.json)

Pass --model-ce to score with the supervised checkpoint
(results/ce_ft/v2m3_pool8_e2/best) instead of the zero-shot reranker.

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

Both intermediate runs (dense, cross-encoder) are cached, and every cache file
carries a fingerprint sidecar over the query file and every parameter that
changes its bytes. A run produced for a different query file or a different
model is therefore refused instead of being silently reused; pass --refresh to
force a rebuild. The fusion configuration and both fingerprints are recorded
next to the output in <out>.config.json, so a submission can be traced back to
the exact inputs that produced it.
"""
import argparse
import hashlib
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
QPREFIX = "Represent this sentence for searching relevant passages: "
CE_Q_MODE, CE_Q_CAP, CE_WIN, CE_POOL = "title", 256, 250, "max"


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_digest(path):
    """Content proxy for a model directory: relpath, size, mtime of every file."""
    h = hashlib.sha1()
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            f = os.path.join(root, name)
            try:
                st = os.stat(f)
            except OSError:
                continue
            h.update(os.path.relpath(f, path).encode("utf-8", "ignore"))
            h.update(b"|")
            h.update(str(st.st_size).encode())
            h.update(b"|")
            h.update(str(st.st_mtime_ns).encode())
            h.update(b"|")
    return h.hexdigest()[:16]


def _fp(*parts):
    """Stable short fingerprint over an ordered list of configuration values."""
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"|")
    return h.hexdigest()[:16]


ADOPT_LEGACY = False  # set from --adopt-legacy; see _cached


def _cached(path, fp, what, refresh, strict=False):
    """True when the cached run can be reused for the current inputs."""
    if refresh or not os.path.isfile(path):
        return False
    meta = path + ".fp.json"
    if not os.path.isfile(meta):
        if strict or not ADOPT_LEGACY:
            raise SystemExit(
                "[cache] %s: %s has no fingerprint sidecar, so it cannot be shown "
                "to match the current inputs; refusing to reuse. Delete the file, "
                "pass --refresh to rebuild, or --adopt-legacy to trust it once."
                % (what, os.path.basename(path)))
        print("[cache] %s: %s has no fingerprint (legacy); adopting it at the "
              "user's request" % (what, os.path.basename(path)))
        _write_fp(path, fp, kind=what, provenance="adopted-legacy")
        return True
    try:
        got = json.load(open(meta, encoding="utf-8")).get("fingerprint")
    except Exception:
        got = None
    if got != fp:
        raise SystemExit(
            "[cache] %s mismatch: %s was built with %s, current inputs give %s; "
            "delete the file or pass --refresh" % (what, path, got, fp))
    print("[cache] %s: fingerprint ok (%s)" % (what, fp))
    return True


def _write_fp(path, fp, **extra):
    d = {"fingerprint": fp}
    d.update(extra)
    with open(path + ".fp.json", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


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
    ap.add_argument("--rrf-wa", type=float, default=DENSE_W, help="dense weight (rrf fusion)")
    ap.add_argument("--rrf-wb", type=float, default=CE_W, help="CE weight (rrf fusion)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore existing dense/CE runs and rebuild them")
    ap.add_argument("--strict-cache", action="store_true",
                    help="refuse a cached run that has no fingerprint sidecar")
    ap.add_argument("--adopt-legacy", action="store_true",
                    help="trust cached runs that have no fingerprint sidecar "
                         "(default: refuse them)")
    ap.add_argument("--tag", default="reteco-mxbai-ce-z")
    args = ap.parse_args()
    global ADOPT_LEGACY
    ADOPT_LEGACY = args.adopt_legacy

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

    q_sha = file_sha1(args.queries)
    dense_fp = _fp("dense", q_sha, os.path.abspath(args.index_data), args.model_dense,
                   _dir_digest(args.model_dense), "mean", QPREFIX, args.topk, "mxbai")

    # 1) dense retrieval over the domain corpus
    if not _cached(dense_run, dense_fp, "dense", args.refresh, args.strict_cache):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_dense_generic.py"),
               "--index-data", args.index_data, "--split", "dev",
               "--model", args.model_dense, "--tag", "mxbai", "--pooling", "mean",
               "--query-prefix", QPREFIX,
               "--work", args.work, "--out", dense_run, "--topk", str(args.topk),
               "--examples", args.queries,
               "--meta", os.path.join(args.work, f"dense_{args.domain}.meta.json")]
        subprocess.run(cmd, check=True)
        _write_fp(dense_run, dense_fp, kind="dense", queries=args.queries, q_sha1=q_sha,
                  model=args.model_dense, index_data=args.index_data, topk=args.topk)

    # 2) cross-encoder rerank of the dense top-100
    ce_fp = _fp("ce", file_sha1(dense_run), os.path.abspath(args.model_ce),
                _dir_digest(args.model_ce), CE_Q_MODE, CE_Q_CAP, CE_WIN, CE_POOL,
                TOP_RERANK)
    if not _cached(ce_run, ce_fp, "ce", args.refresh, args.strict_cache):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_chunk_rerank.py"),
               "--domain", args.domain, "--split", "dev", "--run", dense_run,
               "--k", str(TOP_RERANK), "--model", args.model_ce,
               "--q-mode", CE_Q_MODE, "--q-cap", str(CE_Q_CAP), "--win", str(CE_WIN),
               "--pool", CE_POOL, "--batch", "32", "--out", ce_run,
               "--examples", args.queries,
               "--meta", os.path.join(args.work, f"ce_{args.domain}.meta.json")]
        subprocess.run(cmd, check=True)
        _write_fp(ce_run, ce_fp, kind="ce", dense_run=dense_run, model=args.model_ce,
                  q_mode=CE_Q_MODE, q_cap=CE_Q_CAP, win=CE_WIN, pool=CE_POOL, k=TOP_RERANK)

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
            scores = {d: args.rrf_wa / (RRF_C + rd.get(d, 10 ** 6))
                         + args.rrf_wb / (RRF_C + rc.get(d, 10 ** 6))
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
        # tail filler spaced by rank so a tie-fix step never accumulates
        ranked += [(d, -1e9 - 1e-3 * i) for i, d in enumerate(tail)]
        fused[q] = ranked

    with open(args.out, "w", encoding="utf-8") as f:
        for q in sorted(fused):
            prev = None
            for rank, (docid, score) in enumerate(fused[q], 1):
                # De-tie in the *printed* space: two raw scores closer than
                # 5e-7 still print the same value at 6 decimals. Step by exactly
                # one unit of the sixth decimal so the difference cannot round away.
                val = round(score, 6)
                if prev is not None and val >= prev:
                    val = round(prev - 1e-6, 6)
                prev = val
                f.write(f"{q} Q0 {docid} {rank} {val:.6f} {args.tag}" + chr(10))

    # 4) record the exact configuration next to the run, then check coverage
    cfg = {"out": args.out, "queries": args.queries, "q_sha1": q_sha,
           "dense_fp": dense_fp, "ce_fp": ce_fp,
           "model_dense": args.model_dense, "model_ce": args.model_ce,
           "topk": args.topk, "rerank": TOP_RERANK, "fusion": args.fusion,
           "wa": args.wa, "wb": args.wb,
           "rrf_wa": args.rrf_wa, "rrf_wb": args.rrf_wb,
           "ce_q_mode": CE_Q_MODE, "ce_q_cap": CE_Q_CAP, "ce_win": CE_WIN,
           "ce_pool": CE_POOL, "tag": args.tag}
    with open(args.out + ".config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)

    qids = [r["id"] for r in read_jsonl(args.queries)]
    missing = [q for q in qids if q not in fused]
    cfg["topics"] = len(fused)
    cfg["missing_topics"] = missing
    print(json.dumps(cfg, indent=1))
    if missing:
        sys.exit(f"missing topics: {missing}")


if __name__ == "__main__":
    main()
