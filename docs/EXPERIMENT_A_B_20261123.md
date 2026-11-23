# Experiment A/B (2026-11-23)

## A. Negative-label contamination in CE training

The frozen CE trainer labels every document outside the judged-relevant set as
negative, including documents nobody judged. 62.4% of the dev pool documents the
system ranks wrongly are unjudged, so this could be poisoning training.

Measured (dev domain-macro, identical chain and init):

- frozen recipe: 0.35818
- drop hard negatives with dense rank <= 10: 0.34086 (-0.01732)
- drop all hard negatives (keep dense ranks 21-100 + random): 0.33076 (-0.02743)

Blind sample of 260 hard negatives judged by Qwen3-4B: p >= 0.5 for
86 (33.1%), p >= 0.9 for 76;
mean p by dense rank 1-2 = 0.658, 3-4 = 0.444, 11-20 = 0.241.

Verdict: the labels are noisy, but the noisy documents are the load-bearing ones.
Removal is falsified; only re-weighting remains in scope.

## B. Facet (step-level) retrieval

- 484 dev queries carry decomposition steps.
- Whole-query BM25 finds gold in top-100 for 250;
  of the 234 misses, 113 are recovered by at least one step (48%).
- Union pool (dense top-100 + step BM25): zero-query coverage 86 -> 94,
  42 new judged documents.
- Frozen CE rerank of the union pool: 5 of 15 affected queries become non-zero,
  mean nDCG@10 0.081; ceiling about +0.0015 domain-macro.

Verdict: real but small; not a path to +0.01.

## Artefacts

`results/aggregate_nohard_dev.json`, `results/aggregate_pu10_dev.json`,
`results/analysis/hardneg_probe.json`, `results/analysis/facet_recall_dev.json`,
`results/analysis/facet_pool_dev.json`, `results/analysis/pool_union_dev.json`,
`results/analysis/zero_facet_gain.json`, `results/analysis/zero_facet_ce.json`.
