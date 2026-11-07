"""Shared IO helpers for the RETECO Track 1a Phase-0 experiments.

All paths are explicit; nothing here depends on the working directory.
"""
import json
import os


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_corpus(domain_dir):
    """Return (doc_ids, texts) in file order; documents.jsonl has keys id/content."""
    ids, texts = [], []
    with open(os.path.join(domain_dir, "documents.jsonl"), encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            ids.append(d["id"])
            texts.append(d["content"])
    return ids, texts


def load_examples(domain_dir, split):
    """Track 1a topics: whole query, topic id = id."""
    recs = read_jsonl(os.path.join(domain_dir, f"examples_{split}.jsonl"))
    return [r["id"] for r in recs], [r["query"] for r in recs]


def load_qrels(domain_dir, split):
    """TREC qrels -> {qid: {docid: rel}}, binary relevance in this release."""
    path = os.path.join(domain_dir, f"qrels_{split}.txt")
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            qid, _, docid, rel = ln.split()
            out.setdefault(qid, {})[docid] = int(rel)
    return out


def write_trec_run(path, ranked, tag="run"):
    """ranked: {qid: [(docid, score), ...]} already sorted by descending score."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, docs in ranked.items():
            for rank, (d, s) in enumerate(docs, 1):
                f.write(f"{qid}\tQ0\t{d}\t{rank}\t{s:.6f}\t{tag}\n")


def read_trec_run(path):
    """Return {qid: [(rank, score, docid)]} sorted by rank."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            qid, _, docid, rank, score, _tag = ln.split()
            out.setdefault(qid, []).append((int(rank), float(score), docid))
    for qid in out:
        out[qid].sort(key=lambda r: r[0])
    return out
