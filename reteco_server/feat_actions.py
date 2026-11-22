"""Attach pre-judgement features (dense/CE score and rank of incumbent & challenger,
slot weight, lexical overlap) to every judged action of the widened-space cache."""
import json, math, re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/homeC/wujiaheng/RETECO")
CACHE = ROOT / "results/replacement_v2/expanded3_fixed_dev.json"

def read_run(p):
    out = defaultdict(dict); rank = defaultdict(dict)
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip(): continue
            qid, _, doc, r, s, _ = ln.split()
            out[qid][doc] = float(s); rank[qid][doc] = int(r)
    return out, rank

d = json.load(open(CACHE, encoding="utf-8"))
rows = d["budgets"]["18"]["per_query"]
by_dom = defaultdict(list)
for r in rows: by_dom[r["domain"]].append(r)

WORD = re.compile(r"[a-z0-9]+")
def toks(t): return set(WORD.findall(t.lower()))

docs_cache = {}
def doc_text(dom, docid):
    key = (dom, docid)
    if key in docs_cache: return docs_cache[key]
    # documents.jsonl is large; build a lazy index once per domain
    if dom not in docs_cache:
        idx = {}
        with open(ROOT / f"data/track1_tempo/{dom}/documents.jsonl", encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                idx[o.get("id") or o.get("docid")] = ((o.get("title") or "") + " " + (o.get("content") or o.get("text") or ""))[:2000]
        docs_cache[dom] = idx
    return docs_cache[dom].get(docid, "")

out = []
for dom, qrows in by_dom.items():
    dense, drank = read_run(ROOT / f"runs/dense_mxbai_{dom}_dev.trec")
    ce, crank = read_run(ROOT / f"runs/frozenrepro_{dom}_dev_ce.trec")
    for r in qrows:
        qid = r["qid"]
        qtxt = ""
        for ex in open(ROOT / f"data/track1_tempo/{dom}/examples_dev.jsonl", encoding="utf-8"):
            o = json.loads(ex)
            if str(o.get("qid") or o.get("_id") or o.get("id")) == qid:
                qtxt = str(o.get("query") or ""); break
        qt = toks(qtxt)
        for a in r["actions"]:
            inc, ch = a["incumbent"], a["challenger"]
            di, dc = dense[qid].get(inc), dense[qid].get(ch)
            ei, ec = ce[qid].get(inc), ce[qid].get(ch)
            it = toks(doc_text(dom, inc)); ct = toks(doc_text(dom, ch))
            out.append({
                "domain": dom, "qid": qid, "slot": a["slot"],
                "delta": a["delta"], "mp": a["mp"], "agree": a["agree"], "prior": a["prior"],
                "dense_inc": di, "dense_ch": dc, "ce_inc": ei, "ce_ch": ec,
                "dense_rank_ch": drank[qid].get(ch), "ce_rank_ch": crank[qid].get(ch),
                "ov_inc": len(qt & it) / (len(qt) or 1), "ov_ch": len(qt & ct) / (len(qt) or 1),
            })
json.dump(out, open(ROOT / "results/replacement_v2/ea3_features_dev.json", "w"))
print("rows", len(out))
