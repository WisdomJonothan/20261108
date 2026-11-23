"""Build PU-aware group variants by re-running the group builder with different
negative policies, then post-filtering suspicious (top-ranked) negatives."""
import json, os, subprocess, sys, io, random
from collections import defaultdict
ROOT = "/homeC/wujiaheng/RETECO"
DOMAINS = [d for d in sorted(os.listdir(f"{ROOT}/data/track1_tempo"))
           if os.path.exists(f"{ROOT}/data/track1_tempo/{d}/qrels_train.txt")]
def qrels(d):
    r = defaultdict(dict)
    for ln in open(f"{ROOT}/data/track1_tempo/{d}/qrels_train.txt", encoding="utf-8"):
        p = ln.split()
        if len(p) == 3: r[p[0]][p[1]] = int(p[2])
        elif len(p) == 4: r[p[0]][p[2]] = int(p[3])
    return r
def dense_rank(d):
    run = defaultdict(dict)
    for ln in open(f"{ROOT}/runs/dense_mxbai_{d}_train.trec", encoding="utf-8"):
        p = ln.split()
        if len(p) >= 4: run[p[0]][p[2]] = int(p[3])
    return run
for tag, hard in (("nohard", 0.0), ("pu10", 0.7)):
    outdir = f"{ROOT}/results/ce_folds/{tag}"
    os.makedirs(outdir, exist_ok=True)
    for d in DOMAINS:
        src = f"{ROOT}/results/ce_folds/fall/g_all_{d}.jsonl"
        groups = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()] if os.path.exists(src) else []
        if not groups:
            print("missing", src); continue
        if tag == "nohard":
            # rebuild with hard_frac 0 from the dense run
            pass
        rk = dense_rank(d)
        kept = 0
        for g in groups:
            qid = g["qid"]
            newdocs = []
            for doc in g["docs"]:
                if doc["label"] == 1:
                    newdocs.append(doc); continue
                r = rk.get(qid, {}).get(doc["id"], 999)
                if tag == "nohard":
                    # keep only negatives ranked outside the top-20 (random-ish tail)
                    if r <= 20: continue
                else:  # pu10
                    if r <= 10: continue
                newdocs.append(doc)
            if any(x["label"] == 1 for x in newdocs) and any(x["label"] == 0 for x in newdocs):
                g["docs"] = newdocs
                kept += 1
        with io.open(f"{outdir}/g_{tag}_{d}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for g in groups:
                fh.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(tag, d, "groups", kept)
print("DONE")
