# Why the dev score is 0.3581

*A diagnosis of the frozen dev score. All numbers below are recomputed from
`data/track1_tempo/*/qrels_dev.txt`, the raw TREC runs (`runs/`), and
`results/replacement_v2/candidates_dev.json`; the machine-readable version is
`results/analysis/SCORE_DIAGNOSIS.json` (produced by `results/analysis/score_diagnosis.py`,
CPU only).*

> **Correction note (2026-11-22).** An earlier revision of this document reported that
> "the dense top-100 contains a relevant document for only 8 of the 138 zero queries".
> That was a parsing bug: the readers `results/replacement_v2/zero_recall.py` /
> `zero_depth.py` treated TREC column 3 (the rank) as the score and sorted every list by
> descending rank, so the "top 100" slice was ranks 901-1000. The correct numbers are in
> section 3 and replace the retracted ones; the two scripts and their artefacts
> (`zero_recall_dev.json`, `zero_depth_dev.json`) are withdrawn.

## 1. The metric is dominated by a class of queries that score zero

| | queries | share |
| --- | --- | --- |
| dev queries | 519 | 100% |
| frozen top-10 scores **0** | **138** | **26.6%** |
| queries with no relevant document at all | 0 | 0% |

Every query has at least one judged-relevant document, so a zero is a retrieval failure,
not an unanswerable query. Those 138 queries contribute exactly nothing to the
domain-macro, and because the metric weights the 13 domains equally, the shares are
uneven:

| domain | queries | zero | baseline | weight |
| --- | --- | --- | --- | --- |
| quant | 10 | 5 | 0.1538 | 1/13 |
| law | 11 | 3 | 0.4357 | 1/13 |
| workplace | 11 | 4 | 0.3881 | 1/13 |
| iota | 3 | 0 | 0.8383 | 1/13 |
| monero | 19 | 8 | 0.2069 | 1/13 |
| bitcoin | 30 | 13 | 0.2143 | 1/13 |
| travel | 30 | 12 | 0.2494 | 1/13 |
| cardano | 15 | 4 | 0.2258 | 1/13 |
| economics | 25 | 9 | 0.2512 | 1/13 |
| genealogy | 35 | 10 | 0.4060 | 1/13 |
| politics | 45 | 14 | 0.4290 | 1/13 |
| hsm | 45 | 10 | 0.4651 | 1/13 |
| history | 240 | 46 | 0.3918 | 1/13 |

`quant` is 10 queries and half of them score zero; it carries the same 7.7% of the
metric as `history`, which has 240 queries. A single query in `iota` (3 queries) moves
the macro by up to 0.026.

## 2. If the zero class simply reached its domain's non-zero mean

Replacing every zero with the mean of the non-zero queries *in the same domain* — an
upper bound that requires no new information, only a working first stage — gives
0.3581 -> **0.4985** (+0.1404). The median variant gives 0.4385 (+0.0804). The best
measured local correction (Direct, tail swaps, LLM judging) is +0.0065, i.e. **1.8% of
the zero-class headroom is what the current method captures**.

## 3. The zero class is a first-stage recall failure, not a reranking failure

For the 138 zero queries, the best rank of any judged-relevant document in the dense
mxbai run:

| depth reached | none | <=20 | <=50 | <=100 | <=200 | <=1000 |
| --- | --- | --- | --- | --- | --- | --- |
| zero queries (of 138) | **10** | 30 | 63 | **86** | 103 | 128 |

So for **52 of the 138** queries the dense retriever does not return a relevant document
inside its top-100, and for 10 not even inside the top-1000. A reranker, an LLM judge,
another swap, or a bigger judge model cannot fix that: the document is not in the list.
The other 86 are recoverable by promoting a document that is already in the delivered
top-100 — which is exactly the pool the frozen system hands to the reranker.

Consequences for the ceilings:

| intervention | ceiling | note |
| --- | --- | --- |
| ideal rerank of the delivered top-100 | 0.7446 (+0.3865) | assumes every pool document can be ranked optimally; 86/138 zero queries have pool gold |
| ideal reorder of the same ten returned documents | 0.5036 (+0.1455) | 296/519 queries improvable by position alone |
| the frozen cap-3, tail-slot, one-swap protocol | 0.3769 (+0.0188) | 68/519 queries have a beneficial swap |
| Direct on that protocol | 0.3646 (+0.0065) | 34.6% of the protocol's ceiling |

So 37.6% of the pool headroom is position error (reordering the same ten documents) and
62.4% is wrong-document selection inside the pool.

## 4. What this says about the work done so far

- The bounded-correction line (Direct, +0.0065, CI [+0.0021,+0.0117]) is real but it
  operates on the 381 queries that already retrieve something; with 3 candidates at the
  tail its ceiling is 0.3769, and we captured 34.6% of that.
- Every widening of the action space tested so far (top-slot windows, denser candidate
  pools) stays inside that same 0.3769-style ceiling and moves the macro by ~0.01.
- Therefore the low score is not evidence that the method family is wrong; it is
  evidence that the method family was applied to a small slice of the metric.

## 5. Where the next gain has to come from

1. First-stage recall on the zero class: query decomposition / condition extraction,
   hybrid lexical + dense, or a retriever trained on this corpus (the SFT budget that
   went into the cross-encoder could go into the bi-encoder instead).
2. Promoting the 86 pool-reachable zero queries (a reranking/selection problem, but with
   a much larger headroom than the tail protocol).
3. Reranking and judging generally: the measured same-ten-reorder ceiling is 0.5036 and
   the pool ceiling 0.7446, both far above the 0.3769 the protocol allows.

Artefacts: `results/analysis/SCORE_DIAGNOSIS.json` (distribution, zero class, pool
oracle, ordering-only ceiling, per-domain), `results/analysis/SCORE_DIAGNOSIS_EXTRA.json`,
`results/replacement_v2/candidates_dev.json` (frozen pool).
