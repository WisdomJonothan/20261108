"""Append the negative-label audit (experiment A) and the facet-recall probe
(experiment B) to REPORT.md / README.md / docs, and register the claims."""
import json
from pathlib import Path

REPO = Path(r"D:/RETECO/github_repo")
R = REPO / "results" / "replacement_v2"

report = R / "REPORT.md"
readme = REPO / "README.md"
doc = REPO / "docs" / "EXPERIMENT_A_B_20261123.md"

def frozen():
    d = json.load(open(REPO / "results" / "aggregate_frozenrepro_dev.json", encoding="utf-8"))
    return d["domain_macro"]

def agg(name):
    d = json.load(open(REPO / "results" / f"aggregate_{name}_dev.json", encoding="utf-8"))
    return d["domain_macro"], {x["domain"]: x["ndcg10"] for x in d["domains"]}

base = frozen()
nh, nh_dom = agg("nohard")
pu, pu_dom = agg("pu10")
probe = json.load(open(REPO / "results" / "analysis" / "hardneg_probe.json", encoding="utf-8"))
pool = json.load(open(REPO / "results" / "analysis" / "pool_union_dev.json", encoding="utf-8"))
fr = json.load(open(REPO / "results" / "analysis" / "facet_recall_dev.json", encoding="utf-8"))
zf = json.load(open(REPO / "results" / "analysis" / "zero_facet_ce.json", encoding="utf-8"))

n_probe = probe["n"]
th = probe["thresholds"]
miss = sum(v["query_miss"] for v in fr.values())
fixed = sum(v["query_miss_fixed_by_step"] for v in fr.values())
tot = pool["total"]
ce_fixed = sum(1 for r in zf if r["ce_ndcg10"] > 0)
ce_mean = sum(r["ce_ndcg10"] for r in zf) / len(zf)

section = f"""
## 12. Negative-label audit and facet recall (2026-11-23)

### 12.1 Does treating unlabelled documents as negatives hurt? (experiment A)

The CE is trained with `hard + random` negatives drawn from the dense top-100, and
62.4% of the dev pool documents that the system gets wrong were never judged. If
those unjudged documents were often relevant, the training signal would be
contaminated. Two retrainings test the hypothesis directly, with everything else
(supervision source, init, budget, evaluation chain) held fixed:

| negative set | dev domain-macro | delta |
| --- | --- | --- |
| frozen recipe (hard top-20 + random) | {base:.4f} | - |
| hard negatives with dense rank <= 10 removed | {pu:.4f} | {pu - base:+.4f} |
| no hard negatives (only dense rank 21-100 + random) | {nh:.4f} | {nh - base:+.4f} |

Removing the suspicious negatives makes the CE **worse**, monotonically in how much
is removed (law -0.048, workplace -0.048, hsm -0.034 for the no-hard arm; bitcoin
+0.004 and quant +0.001 are the only gains). The unjudged hard negatives are not
noise to be cleaned away: they are where the cross-encoder learns to separate the
top of the list, and the dev pool is drawn from exactly that band.

A blind model sample of {n_probe} hard negatives (two per query, dense ranks 1-20,
no qrels consulted during sampling) supports the *premise* while contradicting the
*remedy*: a Qwen3-4B judge assigns p(relevant) >= 0.5 to {th['0.5']} of {n_probe}
({100 * th['0.5'] / n_probe:.1f}%) and >= 0.9 to {th['0.9']}, and the rate rises
towards the head of the list (mean p = 0.658 for dense ranks 1-2, 0.241 for ranks
11-20). So the labels are indeed noisy, but the documents carrying that noise are
also the ones the model most needs to see; filtering them out is not a fix. If this
line is pursued at all it must be re-weighting rather than removal.

### 12.2 Does facet retrieval recover the documents the query misses? (experiment B)

The official train/dev splits ship decomposition steps (`steps_dev.jsonl`,
`qrels_steps_dev.txt`) and a per-step relevance file. Using each step as its own
BM25 query over the same corpus, and comparing with the whole-query BM25 retrieval:

| quantity | value |
| --- | --- |
| dev queries with steps | {sum(v['queries'] for v in fr.values())} |
| whole-query BM25 finds a judged document in the top-100 | {sum(v['query_hit'] for v in fr.values())} |
| queries it misses | {miss} |
| of those, at least one step recovers a judged document | {fixed} ({100 * fixed / miss:.0f}%) |

Per domain the effect is concentrated where the corpus is large and the questions are
multi-hop (history {fr['history']['query_miss_fixed_by_step']}/{fr['history']['query_miss']},
genealogy {fr['genealogy']['query_miss_fixed_by_step']}/{fr['genealogy']['query_miss']},
travel {fr['travel']['query_miss_fixed_by_step']}/{fr['travel']['query_miss']}) and absent
for politics/hsm/monero.

At the pool level, however, the dense retriever is much stronger than BM25, so the
union only adds a little: pool coverage of the 138 zero-score queries goes
{tot['zd']} -> {tot['zu']} (+{tot['zu'] - tot['zd']}), and the union adds
{tot['new_gold']} judged-relevant documents that dense never returns. Reranking the
union pool with the frozen cross-encoder converts {ce_fixed} of the 15 queries that
gained a new judged document into a non-zero score (mean nDCG@10 {ce_mean:.3f} over
those 15), which is worth about **+0.0015 domain-macro** at best.

Conclusion: step-level retrieval is a real but small lever. It changes pool coverage
by eight queries and the frozen reranker converts about a third of them, so it cannot
carry a score increase on its own; it is recorded here so the same ground is not
re-walked.
"""

txt = report.read_text(encoding="utf-8")
if "## 12. Negative-label audit and facet recall" not in txt:
    marker = "## 10. Conclusion"
    if marker in txt:
        txt = txt.replace(marker, section.strip() + "\n\n" + marker, 1)
    else:
        txt = txt + "\n" + section
    report.write_text(txt, encoding="utf-8")
    print("REPORT updated")

bullet = f"""
**Negative-label audit and facet recall (2026-11-23).** Removing the unjudged hard
negatives from CE training is *worse*, not better: dev {pu:.4f} when the dense top-10
negatives are dropped and {nh:.4f} when all hard negatives are dropped, against
{base:.4f} for the frozen recipe — the unjudged documents are where the cross-encoder
learns the head of the list, even though a model sample says {100 * th['0.5'] / n_probe:.0f}%
of them look relevant. Step-level (facet) BM25 recovers a judged document for
{fixed} of the {miss} queries that whole-query BM25 misses, but at pool level the
union only lifts zero-query coverage {tot['zd']} -> {tot['zu']} and the frozen
cross-encoder converts {ce_fixed} of the 15 affected queries, i.e. about +0.0015
domain-macro. Details in `results/replacement_v2/REPORT.md` §12.
"""
rt = readme.read_text(encoding="utf-8")
if "Negative-label audit and facet recall (2026-11-23)" not in rt:
    anchor = "**Transfer check (2026-11-23).**"
    if anchor in rt:
        rt = rt.replace(anchor, bullet.strip() + "\n\n" + anchor, 1)
    else:
        rt = rt + "\n" + bullet
    readme.write_text(rt, encoding="utf-8")
    print("README updated")

doc.write_text(f"""# Experiment A/B (2026-11-23)

## A. Negative-label contamination in CE training

The frozen CE trainer labels every document outside the judged-relevant set as
negative, including documents nobody judged. 62.4% of the dev pool documents the
system ranks wrongly are unjudged, so this could be poisoning training.

Measured (dev domain-macro, identical chain and init):

- frozen recipe: {base:.5f}
- drop hard negatives with dense rank <= 10: {pu:.5f} ({pu - base:+.5f})
- drop all hard negatives (keep dense ranks 21-100 + random): {nh:.5f} ({nh - base:+.5f})

Blind sample of {n_probe} hard negatives judged by Qwen3-4B: p >= 0.5 for
{th['0.5']} ({100 * th['0.5'] / n_probe:.1f}%), p >= 0.9 for {th['0.9']};
mean p by dense rank 1-2 = 0.658, 3-4 = 0.444, 11-20 = 0.241.

Verdict: the labels are noisy, but the noisy documents are the load-bearing ones.
Removal is falsified; only re-weighting remains in scope.

## B. Facet (step-level) retrieval

- {sum(v['queries'] for v in fr.values())} dev queries carry decomposition steps.
- Whole-query BM25 finds gold in top-100 for {sum(v['query_hit'] for v in fr.values())};
  of the {miss} misses, {fixed} are recovered by at least one step ({100 * fixed / miss:.0f}%).
- Union pool (dense top-100 + step BM25): zero-query coverage {tot['zd']} -> {tot['zu']},
  {tot['new_gold']} new judged documents.
- Frozen CE rerank of the union pool: {ce_fixed} of 15 affected queries become non-zero,
  mean nDCG@10 {ce_mean:.3f}; ceiling about +0.0015 domain-macro.

Verdict: real but small; not a path to +0.01.

## Artefacts

`results/aggregate_nohard_dev.json`, `results/aggregate_pu10_dev.json`,
`results/analysis/hardneg_probe.json`, `results/analysis/facet_recall_dev.json`,
`results/analysis/facet_pool_dev.json`, `results/analysis/pool_union_dev.json`,
`results/analysis/zero_facet_gain.json`, `results/analysis/zero_facet_ce.json`.
""", encoding="utf-8")
print("docs written")

# claims
claims = REPO / "scripts" / "check_claims.py"
ct = claims.read_text(encoding="utf-8")
block = f'''
    {{
        "name": "CE with no hard negatives",
        "path": "results/aggregate_nohard_dev.json",
        "field": "domain_macro",
        "value": {nh:.5f},
        "tol": 1e-4,
        "in_paper": False,
    }},
    {{
        "name": "CE with dense-top-10 negatives dropped",
        "path": "results/aggregate_pu10_dev.json",
        "field": "domain_macro",
        "value": {pu:.5f},
        "tol": 1e-4,
        "in_paper": False,
    }},
    {{
        "name": "zero queries with a judged document in the dense top-100",
        "path": "results/analysis/pool_union_dev.json",
        "field": "total.zd",
        "value": {tot['zd']},
        "tol": 0,
        "in_paper": False,
    }},
    {{
        "name": "zero queries covered after the facet pool",
        "path": "results/analysis/pool_union_dev.json",
        "field": "total.zu",
        "value": {tot['zu']},
        "tol": 0,
        "in_paper": False,
    }},
'''
if "CE with no hard negatives" not in ct:
    anchor = "CLAIMS = ["
    i = ct.index(anchor) + len(anchor)
    ct = ct[:i] + block + ct[i:]
    claims.write_text(ct, encoding="utf-8")
    print("claims added")
