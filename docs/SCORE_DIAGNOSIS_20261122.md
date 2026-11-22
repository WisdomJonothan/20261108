# Why the dev score is 0.3581

*A diagnosis written before any further experiment. Numbers computed from
`data/track1_tempo/*/qrels_dev.txt`, `results/replacement_v2/candidates_dev.json` and
`results/replacement_v2/zero_recall_dev.json` (CPU, no model involved).*

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
0.3581 → **0.472** (+0.114). The best measured local correction (Direct, tail swaps,
LLM judging) is +0.0065, i.e. **1.8% of the headroom sits in the class the current
method cannot see**.

## 3. The zero class is a first-stage recall failure, not a reranking failure

For the 138 zero queries, the best rank of any judged-relevant document:

| run | none | <=20 | 21-50 | 51-100 |
| --- | --- | --- | --- | --- |
| dense mxbai top-100 | **130** | 2 | 2 | 4 |
| frozen CE rerank (fusion pool) | 52 | 4 | 24 | 58 |

So for 130 of the 138 queries the dense retriever never returns a relevant document in
its top-100. A reranker, an LLM judge, another swap, or a bigger judge model cannot fix
that: the document is not in the list. The cross-encoder's own pool does reach 86 of
them, which means the information is reachable by *some* first stage — just not by this
dense one.

## 4. What this says about the work done so far

- The bounded-correction line (Direct, +0.0065, CI [+0.0021,+0.0117]) is real but it
  operates on the 381 queries that already retrieve something; with 3 candidates at the
  tail its ceiling is 0.3769, and we captured 34.6% of that.
- The wide pool ceilings (0.4098 at top-20, 0.5833 at top-100 with any slot) are *also*
  computed over lists that already contain a relevant document, so they overstate what
  any reranker can do on the zero class.
- Therefore the low score is not evidence that the method family is wrong; it is
  evidence that the method family was applied to the wrong 73% of the metric.

## 5. Where the next gain has to come from

1. First-stage recall on the zero class: query decomposition / condition extraction,
   hybrid lexical + dense, or a retriever trained on this corpus (the SFT budget that
   went into the cross-encoder could go into the bi-encoder instead).
2. Depth: the CE pool already reaches 86 of 138, so fusing a deeper or different
   candidate source into the CE is a cheap prerequisite.
3. Only then does reranking/judging matter — and there the measured ceiling is the
   0.5833 ladder, not 0.3769.

Artefacts: `results/replacement_v2/zero_recall_dev.json` (per query, per run, best rank),
`results/replacement_v2/candidates_dev.json` (frozen pool).
