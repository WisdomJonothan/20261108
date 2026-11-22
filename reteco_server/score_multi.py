"""Score several dev runs under the official domain-macro (13 domains equal weight)
and report the number of zero-scoring queries for each."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/track1_tempo"
RUNS = ROOT / "runs"
DOMAINS = sorted(p.name for p in DATA.iterdir() if p.is_dir() and (p / "qrels_dev.txt").exists())
KS = (10,)


def read_run(path):
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 4:
            out.setdefault(p[0], []).append((p[2], float(p[3])))
    for q in out:
        out[q].sort(key=lambda x: -x[1])
    return out


def read_qrels(path):
    g = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 4 and int(p[3]) > 0:
            g.setdefault(p[0], set()).add(p[2])
    return g


def ndcg10(ranked, rel):
    dcg = 0.0
    for i, doc in enumerate(ranked[:10]):
        if doc in rel:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    return dcg / ideal if ideal > 0 else 0.0


def score(tag, depth=1000):
    per_dom = {}
    zeros = 0
    n = 0
    for d in DOMAINS:
        qrels = read_qrels(DATA / d / "qrels_dev.txt")
        run = read_run(RUNS / tag.format(d=d))
        vals = []
        for q, rel in qrels.items():
            ranked = [doc for doc, _ in run.get(q, [])][:depth]
            v = ndcg10(ranked, rel)
            vals.append(v)
            n += 1
            if v == 0.0:
                zeros += 1
        per_dom[d] = sum(vals) / len(vals) if vals else 0.0
    macro = sum(per_dom.values()) / len(per_dom)
    return macro, zeros, n


def fuse(dense_tag, ce_tag, depth=1000, out_name=None):
    """z-score each run over its own top-100, fuse 1:1, write a run."""
    per_q = {}
    for d in DOMAINS:
        dense = read_run(RUNS / dense_tag.format(d=d))
        ce = read_run(RUNS / ce_tag.format(d=d))
        for q, docs in ce.items():
            dd = dense.get(q, [])
            all_docs = {doc: sc for doc, sc in dd[:depth]}
            for doc, sc in docs:
                all_docs.setdefault(doc, None)
            # z-score dense over top-100
            top = [s for _, s in dd[:100]]
            mu = sum(top) / len(top) if top else 0.0
            sd = (sum((s - mu) ** 2 for s in top) / len(top)) ** 0.5 or 1.0
            cz = [s for _, s in docs]
            cmu = sum(cz) / len(cz) if cz else 0.0
            csd = (sum((s - cmu) ** 2 for s in cz) / len(cz)) ** 0.5 or 1.0
            fused = {}
            for doc, sc in dd[:depth]:
                fused[doc] = (sc - mu) / sd
            for doc, sc in docs:
                fused[doc] = fused.get(doc, 0.0) + (sc - cmu) / csd
            per_q[(d, q)] = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    if out_name:
        for d in DOMAINS:
            with (RUNS / out_name.format(d=d)).open("w", encoding="utf-8") as f:
                for (dd, q), ranked in per_q.items():
                    if dd != d:
                        continue
                    for i, (doc, sc) in enumerate(ranked[:1000], 1):
                        f.write(f"{q}\tQ0\t{doc}\t{i}\t{sc:.6f}\t{out_name.format(d=d).replace('_dev.trec','')}\n")
    per_dom = {}
    zeros = 0
    n = 0
    for d in DOMAINS:
        qrels = read_qrels(DATA / d / "qrels_dev.txt")
        vals = []
        for q, rel in qrels.items():
            ranked = [doc for doc, _ in per_q.get((d, q), [])][:10]
            v = ndcg10(ranked, rel)
            vals.append(v)
            n += 1
            if v == 0.0:
                zeros += 1
        per_dom[d] = sum(vals) / len(vals) if vals else 0.0
    return sum(per_dom.values()) / len(per_dom), zeros, n


def main():
    tests = [
        ("frozen sub_sft_e2bc", "sub_sft_e2bc_{d}_dev.trec", 1000),
        ("dense mxbai top-10", "dense_mxbai_{d}_dev.trec", 10),
        ("ce-only full_ce_s17", "fusion100_full_ce_s17_20261115_{d}_dev_ce.trec", 10),
        ("sub_sft_e2bc top100 (run stores 1000?)", "sub_sft_e2bc_{d}_dev.trec", 100),
    ]
    for name, tag, depth in tests:
        m, z, n = score(tag, depth)
        print(f"{name:44s} macro={m:.4f} zeros={z}/{n}")
    m, z, n = fuse("dense_mxbai_{d}_dev.trec", "fusion100_full_ce_s17_20261115_{d}_dev_ce.trec", 1000, "deepfuse_{d}_dev.trec")
    print(f"{'fuse(dense1000, ce100) z1:1':44s} macro={m:.4f} zeros={z}/{n}")


if __name__ == "__main__":
    main()
