# Constraint-loss subsets on dev (519 queries)

Split by `results/QUERY_CONSTRAINT_FLAGS.json`: a query is `lost` when at least
one year / temporal / exclusion / comparison cue appears in the body but not
in the title (241 queries), `kept` otherwise (278). Domain-macro nDCG@10.

| system | lost (241) | kept (278) | kept - lost |
|---|---|---|---|
| dense mxbai | 0.3044 | 0.2892 | -0.0152 |
| CE only (SFT 2ep) | 0.2789 | 0.2759 | -0.0029 |
| CE only (SFT 2ep) gain vs previous | -0.0256 | -0.0133 | |
| +zero-shot CE, z 1:1 | 0.3175 | 0.3272 | +0.0097 |
| +zero-shot CE, z 1:1 gain vs previous | +0.0386 | +0.0513 | |
| +SFT CE, z 1:1 (frozen) | 0.3656 | 0.3518 | -0.0139 |
| +SFT CE, z 1:1 (frozen) gain vs previous | +0.0481 | +0.0246 | |

Reading: the frozen system is *better* on the constraint-lost subset (+0.0138),
and the supervised gain over zero-shot CE is roughly twice as large there
(+0.0481 vs +0.0246) even though neither the dense retriever nor the reranker
ever sees the body. So on these queries the body conditions are not what the
ranker is missing: with a title alone it still separates better than on the
other half. That weakens the naive 'truncation loses the answer' story and is
exactly why the `ctxt` arm is pre-registered to show its gain *concentrated* on
the lost subset (docs/RESEARCH_PLAN_NEXT.md 4b) rather than assumed.
Caveat: the subsets differ in domain mix (history contributes 343/775 lost).
