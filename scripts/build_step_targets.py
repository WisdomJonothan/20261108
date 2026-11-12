#!/usr/bin/env python3
"""Build per-query STEP -> gold-document targets for step-supervised ranking.

The organisers annotate every TRAIN question with the ordered evidence steps a
complete answer needs, and give qrels for each step
(`data/track1_tempo/<domain>/steps_train.jsonl` + `qrels_steps_train.txt`).
Track 1a only ever sees the whole question, so the steps are used here as a
*training* signal: a candidate group should put at least one document for every
required step in the top of its ranking, not just one easy document that
happens to support the most obvious step.

Writes `results/ce_step_targets_<domain>.json`:
    {qid: [[doc_id, ...], [doc_id, ...], ...]}   # one list per step

Coverage measured on 2026-11-10: 91.1% of steps (2321/2547 inside the built
candidate groups) have at least one gold document inside the group, so the
signal survives the dense top-100 truncation for most steps.
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "track1_tempo")


def main():
    total_q = total_s = 0
    for dom_dir in sorted(glob.glob(os.path.join(DATA, "*"))):
        domain = os.path.basename(dom_dir)
        steps_path = os.path.join(dom_dir, "steps_train.jsonl")
        qrels_path = os.path.join(dom_dir, "qrels_steps_train.txt")
        if not (os.path.exists(steps_path) and os.path.exists(qrels_path)):
            print(f"skip {domain}: no step files")
            continue
        step_docs = {}
        with open(qrels_path, encoding="utf-8") as f:
            for ln in f:
                parts = ln.split()
                if len(parts) != 4:
                    continue
                sid, _, doc, rel = parts
                if int(rel) > 0:
                    step_docs.setdefault(sid, []).append(doc)
        out, n_steps = {}, 0
        with open(steps_path, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                qid = r.get("id")
                step_ids = [s.get("step_id") for s in (r.get("steps") or [])]
                step_ids = [s for s in step_ids if s]
                if not qid or not step_ids:
                    continue
                targets = []
                for sid in step_ids:
                    docs = sorted(set(step_docs.get(sid, [])))
                    if docs:          # steps with no judged document carry no signal
                        targets.append(docs)
                if targets:
                    out[qid] = targets
                    n_steps += len(targets)
        dst = os.path.join(ROOT, "results", f"ce_step_targets_{domain}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f)
        total_q += len(out)
        total_s += n_steps
        print(f"{domain:10s} queries={len(out):4d} steps_with_docs={n_steps:5d} -> {dst}")
    print(f"total queries={total_q} steps={total_s}")


if __name__ == "__main__":
    main()
