"""Identify which CE run the frozen submission was fused from (bitcoin probe)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import argparse
p = argparse.ArgumentParser()
p.add_argument("--domain", default="bitcoin")
a = p.parse_args()

import ir_metrics  # noqa: E402


def read_run(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        qid, docid = parts[0], parts[2]
        score = float(parts[4]) if len(parts) >= 6 else float(parts[3])
        out.setdefault(qid, []).append((docid, score))
    for v in out.values():
        v.sort(key=lambda x: -x[1])
    return out


def read_qrels(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        qid = parts[0]
        docid = parts[1] if len(parts) == 3 else parts[2]
        g = int(float(parts[-1]))
        if g > 0:
            out.setdefault(qid, set()).add(docid)
    return out


def ndcg(ranked, gold):
    dcg = sum(1.0 / __import__("math").log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
    idcg = sum(1.0 / __import__("math").log2(i + 2) for i in range(min(10, len(gold))))
    return dcg / idcg if idcg else 0.0


dom = a.domain
qrels = read_qrels(ROOT / "data" / "track1_tempo" / dom / "qrels_dev.txt")
frozen = read_run(ROOT / "runs" / f"sub_sft_e2bc_{dom}_dev.trec")
cands = sorted(ROOT.glob(f"runs/*_{dom}_dev_fused.trec"))
cands += sorted(ROOT.glob(f"runs/*_{dom}_dev_calibrated.trec"))
print(f"frozen top1 sample: {[d for d, _ in frozen[list(frozen)[0]][:3]]}")
for path in cands:
    run = read_run(path)
    common = set(run) & set(frozen)
    if not common:
        continue
    same10 = sum(1 for q in common if [d for d, _ in run[q][:10]] == [d for d, _ in frozen[q][:10]])
    scores = [ndcg([d for d, _ in run[q][:10]], qrels.get(q, set())) for q in common]
    print(f"{path.name:65s} same-top10 {same10}/{len(common)}  nDCG {sum(scores)/len(scores):.4f}")
