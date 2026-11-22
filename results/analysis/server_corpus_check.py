import json, glob, os

root = os.path.expanduser("~/RETECO/data/track1_tempo")
doms = sorted(os.path.basename(p) for p in glob.glob(root + "/*") if os.path.isdir(os.path.join(p)))
res = {}
for d in doms:
    qf = os.path.join(root, d, "qrels_dev.txt")
    cf = os.path.join(root, d, "documents.jsonl")
    if not (os.path.exists(qf) and os.path.exists(cf)):
        continue
    gold = {}
    for ln in open(qf):
        p = ln.split()
        if len(p) >= 4 and int(p[3]) > 0:
            gold.setdefault(p[0], set()).add(p[2])
    need = set().union(*gold.values()) if gold else set()
    found = set()
    n = 0
    keys = None
    with open(cf) as f:
        for ln in f:
            n += 1
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if keys is None:
                keys = sorted(o.keys())
            did = o.get("id") or o.get("docid") or o.get("_id")
            if did in need:
                found.add(did)
                if len(found) == len(need):
                    break
    res[d] = dict(corpus_docs_scanned=n, keys=keys, qrel_docids=len(need), found=len(found),
                  missing=sorted(need - found)[:5], queries=len(gold),
                  gold_counts=sorted(len(v) for v in gold.values()))
print(json.dumps(res, indent=0)[:6000])
