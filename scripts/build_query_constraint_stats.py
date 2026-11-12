#!/usr/bin/env python
"""Cue statistics over the Track-1 train+dev queries, plus per-query flags.

The point: a query's *title* is what a title-only reranker sees, while the body
often carries the time range / exclusion / comparison that makes the question
answerable. This script counts how often a cue exists in the body but not in the
title, and writes a per-query flag file so downstream evaluation can split by it.

Definition (v1, broad -- counts text that *mentions* a cue):
  year       \b1[5-9][0-9][0-9]\b | \b20[0-9][0-9]\b
  temporal   before|after|between|during|since|until|prior to
  exclusion  "not " | exclud | except | without | other than
  comparison more than|less than|fewer|at least|at most|than|first|last|earliest|latest

Outputs
  results/QUERY_CONSTRAINT_STATS.json   aggregates (regenerated, must match docs)
  results/QUERY_CONSTRAINT_FLAGS.json   {qid: {domain, split, title, cues_lost}}
"""
import glob
import io
import json
import os
import re
import sys

CUES = {
    "year": re.compile(r"\b1[5-9][0-9][0-9]\b|\b20[0-9][0-9]\b"),
    "temporal": re.compile(r"\b(before|after|between|during|since|until|prior to)\b", re.I),
    "exclusion": re.compile(r"(not |exclud|except|without|other than)", re.I),
    "comparison": re.compile(r"(more than|less than|fewer|at least|at most|than|first|last|earliest|latest)", re.I),
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def split_raw(raw):
    parts = (raw or "").split("<p>")
    return parts[0], (parts[1] if len(parts) > 1 else "")


def clean(t):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t or "")).strip()


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data/track1_tempo/*/examples_*.jsonl"))):
        domain = path.replace("\\", "/").split("/")[-2]
        split = "dev" if "_dev.jsonl" in path else "train"
        for line in io.open(path, encoding="utf-8"):
            r = json.loads(line)
            title, body = split_raw(r.get("query") or r.get("text") or "")
            title, body = clean(title), clean(body)
            lost = []
            for name, rx in CUES.items():
                if rx.search(body) and not rx.search(title):
                    lost.append(name)
            rows.append({"qid": str(r["id"]), "domain": domain, "split": split,
                         "cues_lost": lost})

    stats = {"n_queries": len(rows), "cues": {}, "any_lost": 0, "per_domain_any_lost": {}}
    for name in CUES:
        body = sum(1 for r in rows if CUES[name].search(r.get("_body", "")))  # unused
    # counts are recomputed below from the same rows for clarity
    body_hits = {k: 0 for k in CUES}
    title_hits = {k: 0 for k in CUES}
    lost_hits = {k: 0 for k in CUES}
    for r in rows:
        for name in r["cues_lost"]:
            lost_hits[name] += 1
        if r["cues_lost"]:
            stats["any_lost"] += 1
            stats["per_domain_any_lost"][r["domain"]] = \
                stats["per_domain_any_lost"].get(r["domain"], 0) + 1
    # body/title totals need the texts again: cheaper to recount in one pass below
    for path in sorted(glob.glob(os.path.join(ROOT, "data/track1_tempo/*/examples_*.jsonl"))):
        for line in io.open(path, encoding="utf-8"):
            r = json.loads(line)
            title, body = split_raw(r.get("query") or r.get("text") or "")
            title, body = clean(title), clean(body)
            for name, rx in CUES.items():
                if rx.search(body):
                    body_hits[name] += 1
                if rx.search(title):
                    title_hits[name] += 1
    for name in CUES:
        n = stats["n_queries"]
        stats["cues"][name] = {
            "body": body_hits[name], "title": title_hits[name],
            "lost_by_title": lost_hits[name],
            "pct_lost": round(100.0 * lost_hits[name] / n, 1),
        }
    stats["any_lost_pct"] = round(100.0 * stats["any_lost"] / stats["n_queries"], 1)

    flags = {r["qid"]: {"domain": r["domain"], "split": r["split"],
                        "cues_lost": r["cues_lost"], "any_lost": bool(r["cues_lost"])}
             for r in rows}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(stats, io.open(os.path.join(ROOT, "results/QUERY_CONSTRAINT_STATS.json"), "w",
                             encoding="utf-8"), indent=1)
    json.dump(flags, io.open(os.path.join(ROOT, "results/QUERY_CONSTRAINT_FLAGS.json"), "w",
                             encoding="utf-8"), indent=0)
    print(json.dumps({"n": stats["n_queries"], "any_lost": stats["any_lost"],
                      "pct": stats["any_lost_pct"],
                      "cues": {k: (v["body"], v["title"], v["lost_by_title"])
                               for k, v in stats["cues"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
