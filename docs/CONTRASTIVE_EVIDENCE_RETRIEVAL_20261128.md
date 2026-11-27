# Retrieve the difference between missing and already represented evidence

The [candidate audit](COVERAGE_ROUTED_ADMISSION_20261127.md) found 293 of 883 questions with zero judged positives in the current ten. The [earlier feedback experiment](TEMPORAL_CONFLICT_RERANKING_20261126.md#search-the-corpus-for-evidence-missing-from-the-current-top-results) used existing results to select a literal query fragment and searched from that fragment. The present experiment asks whether the search should explicitly discount what the existing list already represents.

[IRCoT](https://aclanthology.org/2023.acl-long.557/) uses intermediate reasoning to decide what to retrieve next. [GAR](https://arxiv.org/abs/2208.08942) expands candidates via document neighbors. [DIFF](https://arxiv.org/abs/2610.29464) formulates directional feedback as a regularized query update in tangent space. Our comparison measures a specific feedback direction for multi-evidence questions. It contrasts two literal requests from the same query using the current list as the coverage reference, and checks acquisition of newly covered annotated steps at a fixed budget.

## Protocol fixed before scoring

Use the archived 186 and 155 training-side panels and their existing 60-document pools. Keep the current final ten for measuring existing coverage. Both panels have project exposure. Search the original full corpus without changing the index. Generate up to six literal question fragments, keeping each interrogative sentence with its explicit temporal wording. Prepend the first interrogative sentence to each later one as shared context. Abstain when fewer than two fragments survive. Use the original mxbai full-query vector and the same encoder for all fragments.

Estimate a fragment's coverage as its maximum cosine similarity to one of the existing ten documents. Let *g* be the least-covered fragment vector and *c* the most-covered. Keep the original query *q* in every search. The gap-only arm adds the tangent component of *g*. The contrast arm adds the tangent component of *g − c*. The reversal control uses *c − g*. The tangent component removes the part parallel to *q*. Each nonzero correction has norm 0.30 before renormalization, so the controls move by the same angle. Ties or a degenerate direction use the original query. The original query is a fourth control.

Each arm proposes the top four documents outside its unchanged pool of 60 after a fresh full-domain search. Freeze the ranked IDs and fingerprint the manifest before opening training judgments. Report newly reached documents, total queries with at least one relevant offer, annotated steps reached beyond the current ten, and ideal nDCG@10 on the 60 plus four documents. Also report the ideal score of at most one rank-ten admission into the current list. These label-assisted outcomes diagnose the candidate stage. A downstream ranking comparison is warranted only if the contrast improves the candidate oracle over gap-only on both panels and covers more missing steps on the pooled annotated subset.

## First result

| Four-document proposal | 186 candidate oracle | 155 candidate oracle | Newly reached positives, pooled | New steps beyond the current ten, pooled |
| --- | ---: | ---: | ---: | ---: |
| Original query | 0.68289 | 0.68020 | 11 | 7 |
| Gap only | 0.68347 | 0.67828 | 8 | 6 |
| Gap minus covered | 0.68439 | 0.67686 | 11 | 8 |
| Covered minus gap | 0.68692 | 0.67351 | 15 | 9 |

The original-query replay reproduces 99.985% of the original dense top 100. Of the 341 queries, 262 activate the contrast, and 257 of those get a different ordered offer from gap-only. The matched contrast increases the 186 candidate oracle by +0.00092 and decreases the 155 candidate oracle by −0.00143. The sign reversal finds more new positives on each panel. This weakens the assumption that higher similarity to an existing document identifies evidence already supplied, and suggests separating directional exploration from confidence in the coverage proxy. The one-sided contrast stops at candidate evaluation. [Fixed candidates and scores](../results/analysis/contrastive_evidence_retrieval_341_20261128.json.gz)

The forward contrast can find distinct evidence. On economics question `60289_158`, it retrieves a document annotated for critiques of international finance organizations' poverty-reduction strategies. The current ten and the four original-query offers leave that step uncovered.

## Independent two-sided search protocol

Hold out the separate 191-topic archive across eight domains for this test. Use the same original ranking, 60-document pool, encoder, query fragments, and perturbation radius. Alternate the top distinct documents from the forward and reverse perturbations until filling four slots. As a matched two-probe control, alternate original-query and gap-only offers. In the absence of overlap, either allocation takes two from each query. Retain all four single-probe outcomes for interpretation. No new threshold or evidence selection is fitted. The 191 have earlier project exposure and have not been used to develop this two-sided rule.

Freeze the offers prior to scoring. Advance to a fixed joint reranking comparison only if two-sided search increases the full-pool oracle and the number of newly covered steps against both the original-query control and the matched two-probe control.

| Four-document proposal on the 191 | Candidate oracle | New judged positives | Newly covered steps |
| --- | ---: | ---: | ---: |
| Original query | 0.61080 | 9 | 6 |
| Gap only | 0.61209 | 7 | 6 |
| Forward contrast | 0.61273 | 8 | 5 |
| Reverse contrast | 0.60902 | 8 | 4 |
| Original and gap interleaved | 0.60982 | 8 | 5 |
| Forward and reverse interleaved | 0.61170 | **10** | **8** |

The two-sided rule clears the registered candidate gate. Its candidate-oracle changes are +0.00090 [−0.00229, +0.00456] versus the original query and +0.00188 [−0.00174, +0.00609] versus the matched mixture. Of its ten judged positives, four occur in the forward list only, four in the reverse list only, and two in both lists. Forward contrast alone has the highest candidate oracle. The two-sided policy reaches more annotated steps. [Reserve candidate manifest and independent evaluation](../results/analysis/contrastive_evidence_retrieval_191_20261128.json.gz)

## Fixed rank-ten admission

Compare the two-sided proposals with the original-query offers and the matched two-probe offers. Use the same five available documents per arm, four offers and the old rank-ten result. Score all inputs with the pinned Mixedbread cross-encoder and the frozen MiniLM checkpoint using the same dispersed evidence and an identical full-question input procedure. Reuse identical question-document scores across arms.

Replace rank ten only if both rankers prefer the very same new document over all other choices, and each assigns it a strictly higher score than to the incumbent. Keep the remaining nine positions. The candidate IDs and scoring contexts are preserved from the registered comparison, and the ranking record is hashed before measuring final domain-macro nDCG and restored or lost evidence steps. This fixed-budget replay uses the same 191 questions as a downstream diagnostic.

| Final rule on the 191 | Domain-macro nDCG@10 | Rank-ten changes | Gained steps | Lost steps |
| --- | ---: | ---: | ---: | ---: |
| Maintained ranking | 0.28212 | 0 | 0 | 0 |
| Original-query offers | 0.28212 | 26 | 0 | 0 |
| Original and gap mixture | 0.28212 | 29 | 0 | 0 |
| Two-sided offers | 0.28212 | 26 | 0 | 0 |

Each final metric is identical for every query across the four arms. Among the ten two-sided queries with an offered positive, the two rankers put a judged positive first among the offers on four and two queries. Once compared against the original rank ten, the counts fall to two and zero, respectively. The textual admission step therefore loses all of the extra judged positives exposed by the new proposal mechanism. A descriptive replay using only the large scorer yields the same +0.00158 for original-query and two-sided offers. Each makes two helpful and three harmful swaps. This replay was inspected after the registered comparison and is outside model selection.

The full-pool search changes which evidence the system can access. Realizing that opportunity now depends on grading the candidate for the specific unanswered request and its required time scope. An evidence-aware admission test can use the preserved offers and compare that narrower judgment to matched full-question grading. The official dev reference is unchanged. [Final rankings, raw logits and paired outcomes](../results/analysis/contrastive_evidence_admission_191_20261128.json.gz)

## Replay

```text
python scripts/eval_contrastive_evidence_retrieval.py prepare --archives results/analysis --data-root data/track1_tempo --embeddings emb --runs runs --encoder models/mxbai-embed-large-v1 --output offers.json.gz
python scripts/eval_contrastive_evidence_retrieval.py score --manifest offers.json.gz --fingerprint <prepare-output-sha256> --data-root data/track1_tempo --output evaluated.json.gz
```

Pass `--reserve` to prepare the independent 191-query comparison and use new output paths. Both stages use only the original benchmark corpus and training split.

For the fixed final comparison, pass the evaluated reserve candidates and the original reserve archive to `replay_contrastive_evidence_admission.py prepare`. Score its inputs with `eval_fifth_ranker.py score` once per checkpoint, then pass both caches as `--large` and `--small` to `replay_contrastive_evidence_admission.py score`. Use batch sizes 16 for Mixedbread and 32 for MiniLM.
