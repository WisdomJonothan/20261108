#!/usr/bin/env python3
"""Strict submission validator for RETECO Track-1a run files.

Wraps the official `scripts/official/format_checker.py` (same rules as the
CodaBench checker) and adds the checks that only bite at submission time:

  * topic-id set is *exactly* the set in `examples_<split>.jsonl`
    (a missing query scores zero officially, an unknown one is discarded)
  * every document id belongs to that domain's `documents.jsonl`
  * ranks are contiguous 1..n, scores are finite and strictly non-increasing
  * no duplicate doc ids within a topic
  * optional --detie OUT: rewrite a de-tied copy (score - 1e-9*rank) so that a
    score-sorting scorer cannot permute tied top-10 slots

Usage:
  python validate_submission.py --runs "runs/fused_1_1_*_dev.trec" --split dev
  python validate_submission.py --runs runs/x_test.trec --split test \
      --examples-pattern data/track1_tempo/{domain}/examples_test.jsonl
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.environ.get("RETEco_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFICIAL = os.path.join(ROOT, "scripts", "official", "format_checker.py")


DOMAINS = ["bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
           "iota", "law", "monero", "politics", "quant", "travel", "workplace"]


def domain_of(run_path):
    """Resolve the domain from the file name, or refuse.

    An unrecognised name used to fall back to the second-to-last token, which
    silently produced a domain that does not exist: every check then skipped and
    the run was reported clean. That is worse than failing, so an unknown name
    is an error the caller must see.
    """
    base = os.path.basename(run_path)
    if not base.endswith(".trec"):
        raise SystemExit("[validate] %s: not a .trec file" % base)
    stem = base[:-5]
    split = stem.rsplit("_", 1)[-1]
    if split not in ("dev", "test", "train"):
        raise SystemExit(
            "[validate] %s: cannot tell the split from the file name (expected "
            "_dev/_test/_train)" % base)
    for d in DOMAINS:
        if re.search(r"(?:^|_)" + re.escape(d) + r"(?:_|$)", stem[:-len(split) - 1]):
            return d, split
    raise SystemExit(
        "[validate] %s: cannot resolve a Track 1a domain from the file name; "
        "rename it as <anything>_<domain>_<split>.trec"
        % base)


def read_topics(path):
    ids = set()
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                ids.add(json.loads(ln)["id"])
    return ids


def read_corpus_ids(path):
    ids = set()
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                ids.add(json.loads(ln)["id"])
    return ids


def check_run(path, split, examples_pattern, detie):
    domain, run_split = domain_of(path)
    errors, warnings = [], []

    # official checker
    data_dir = os.path.join(ROOT, "data", "track1_tempo", domain)
    qrels = os.path.join(data_dir, f"qrels_{split}.txt")
    corpus = os.path.join(data_dir, "documents.jsonl")
    for need in (qrels, corpus):
        if not os.path.isfile(need):
            raise SystemExit(
                "[validate] %s: missing %s, so this run cannot be checked "
                "(refusing to report it as clean)" % (path, need))
    cmd = [sys.executable, OFFICIAL, path]
    if os.path.exists(qrels):
        cmd += ["--qrels", qrels]
    if os.path.exists(corpus):
        cmd += ["--corpus", corpus]
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        errors.append(f"official_checker rc={p.returncode}: {(p.stdout + p.stderr)[:400]}")

    # parse run
    topics = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 6:
                errors.append(f"bad column count: {ln[:80]}")
                continue
            qid, _, docid, rank, score, _tag = parts
            topics.setdefault(qid, []).append((int(rank), float(score), docid))

    examples = examples_pattern.format(domain=domain, split=split)
    if not os.path.exists(examples):
        candidates = [os.path.join(data_dir, f"examples_{split}.jsonl"),
                      os.path.join(data_dir, f"examples_{run_split}.jsonl")]
        examples = next((c for c in candidates if os.path.exists(c)), examples)
    if examples and os.path.exists(examples):
        want = read_topics(examples)
        got = set(topics)
        if got - want:
            errors.append(f"unknown topic ids: {sorted(got - want)[:5]} (+{len(got - want)})")
        if want - got:
            errors.append(f"missing topics ({len(want - got)}): {sorted(want - got)[:5]} "
                          f"-> these score 0 officially")
    else:
        warnings.append(f"no examples file found for domain={domain} split={split}")

    if os.path.exists(corpus):
        corpus_ids = read_corpus_ids(corpus)
        bad = {d for _, _, d in (x for v in topics.values() for x in v)} - corpus_ids
        if bad:
            errors.append(f"{len(bad)} doc ids not in corpus, e.g. {sorted(bad)[:3]}")

    tie_topics = 0
    tied_top10 = 0
    for qid, rows in topics.items():
        rows.sort(key=lambda r: r[0])
        ranks = [r[0] for r in rows]
        if ranks != list(range(1, len(rows) + 1)):
            errors.append(f"{qid}: ranks not contiguous 1..n")
        docs = [r[2] for r in rows]
        if len(set(docs)) != len(docs):
            errors.append(f"{qid}: duplicate doc ids")
        scores = [r[1] for r in rows]
        if any(s != s or s in (float("inf"), float("-inf")) for s in scores):
            errors.append(f"{qid}: non-finite score")
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            errors.append(f"{qid}: scores increase with rank")
        if len(set(scores)) != len(scores):
            tie_topics += 1
            if len(set(scores[:10])) != len(scores[:10]):
                tied_top10 += 1

    if detie and tie_topics:
        out = detie.format(domain=domain, split=run_split)
        # ordinal de-tie: only break exact ties, keep the rank order the source run
        # already has.  A fixed 1e-9*rank offset is invisible next to the -1e9
        # padding sentinel, so the step must scale with the score magnitude.
        fixed = {}
        for qid, rows in topics.items():
            prev = None
            out_rows = []
            for rank, score, docid in rows:
                if prev is not None and score >= prev:
                    score = prev - max(1e-9, abs(prev) * 1e-9)
                prev = score
                out_rows.append((rank, score, docid))
            fixed[qid] = out_rows
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for qid, rows in fixed.items():
                for rank, score, docid in rows:
                    f.write(f"{qid}	Q0	{docid}	{rank}	{score:.9f}	rrf\n")
        reparsed = {}
        with open(out, encoding="utf-8") as f:
            for ln in f:
                qid, _q0, docid, rank, score, _tag = ln.split()
                reparsed.setdefault(qid, []).append((int(rank), float(score), docid))
        if set(reparsed) != set(topics):
            errors.append("de-tied file: topic set differs from the source run")
        for qid, rows2 in reparsed.items():
            rows2.sort(key=lambda r: r[0])
            sc = [r[1] for r in rows2]
            if any(s != s for s in sc):
                errors.append("de-tied " + qid + ": non-finite score")
                break
            if any(sc[i] <= sc[i + 1] for i in range(len(sc) - 1)):
                errors.append("de-tied " + qid + ": scores not strictly decreasing")
                break
        warnings.append("de-tied copy written and re-validated: " + out)

    return {"run": os.path.relpath(path, ROOT), "domain": domain, "split": run_split,
            "topics": len(topics), "lines": sum(len(v) for v in topics.values()),
            "topics_with_ties": tie_topics, "topics_with_tied_top10": tied_top10,
            "errors": errors, "warnings": warnings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="glob(s) or paths")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--examples-pattern", default="data/track1_tempo/{domain}/examples_{split}.jsonl")
    ap.add_argument("--detie", default=None, help="output pattern, e.g. 'runs/detied_{domain}_{split}.trec'")
    ap.add_argument("--json", default="results/submission_validation.json")
    args = ap.parse_args()

    paths = []
    for pat in args.runs:
        paths.extend(sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat])
    if not paths:
        print("no run files matched", file=sys.stderr)
        return 2
    reports = [check_run(p, args.split, args.examples_pattern, args.detie) for p in paths]
    n_err = sum(len(r["errors"]) for r in reports)
    for r in reports:
        status = "OK " if not r["errors"] else "FAIL"
        print(f"{status} {r['run']:44s} topics={r['topics']:4d} lines={r['lines']:6d} "
              f"ties={r['topics_with_ties']:3d} tied_top10={r['topics_with_tied_top10']:3d}")
        for e in r["errors"][:3]:
            print("   !", e)
        for w in r["warnings"][:2]:
            print("   ~", w)
    out = os.path.join(ROOT, args.json)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "files": reports, "total_errors": n_err}, f, indent=1)
    print(f"\n{len(reports)} files, {n_err} errors -> {out}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
