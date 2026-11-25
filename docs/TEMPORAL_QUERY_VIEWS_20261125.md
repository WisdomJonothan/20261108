# Retrieve separate evidence with the original query text

The current 519-query official dev reference is 0.39589. This study changes candidate selection on 186 held-out training queries and keeps the first 30 candidates unchanged. The additional 30 slots are the only part of the pool affected by the retrieval experiment. Candidate ceilings use official qrels and ideal ordering, while the measured rankings use model scores. [Reference and matched 60-document controls](QWEN_ADAPTATION_AND_PAIR_EVIDENCE_20261125.md)

## Use two views of a question

Generate two complementary evidence searches with the local Qwen3-4B-Instruct-2507 checkpoint. The first procedure writes two short search phrases. It produces two parseable views for 185 of 186 questions. A string audit finds four-digit year-like tokens absent from the source in 75 questions. Manual inspection finds examples that add Bitcoin Core versions and dates to general questions about node operation. Treat this as a direct generation control.

The extractive procedure asks for the numbers of up to three parts already present in the question for each search. Construct each view from those parts and prepend the original first part. This yields two parseable views for 180 questions and preserves the source vocabulary. Malformed outputs use the unchanged original question. Keep the original question as a third view in both procedures.

Encode the three views with the same mxbai encoder, masked mean pooling and 512-token limit as the original first stage. The original-query replay recovers 99.94% of the archived mxbai top 100. Compare adding 30 documents from the original query alone, reciprocal-rank fusion of the views, or taking turns from each ranked view with duplicate documents removed. The existing 60-document comparator draws the extra candidates from the prior frozen fusion, deep mxbai ranks and BGE. [Generated-view query records](../results/analysis/generated_temporal_view_oracle_20261125.json) · [Extractive-view query records](../results/analysis/extractive_temporal_view_oracle_20261125.json) · [Candidate manifests](../results/analysis/temporal_view_candidate_manifests_20261125.tar.gz)

| Training-holdout candidate policy | Ideal nDCG@10 | No positive in pool | Judged positives added past 30 |
| --- | ---: | ---: | ---: |
| Existing 60-document pool | 0.67516 | 31 | 60 |
| Original query alone | 0.68243 | 30 | 66 |
| Generated views, reciprocal-rank fusion | 0.68395 | 28 | 65 |
| Generated views, interleaved | 0.68888 | 28 | 65 |
| Extractive views, reciprocal-rank fusion | 0.68138 | 29 | 68 |
| Extractive views, interleaved | **0.69182** | 29 | **73** |

Interleaving extractive views raises the ceiling by +0.01667 [−0.00739, +0.04040]. It adds 13 judged relevant documents compared with the existing 60 and seven compared with original-query retrieval. The intervals describe variation across the paired training queries.

## Measure ranking with the expanded pools

Score the interleaved generated candidates using identical question and evidence inputs to the earlier models. The protected 60-document policy scores 0.32270 with Qwen3-Reranker-8B and 0.32147 with Yuanbao-Reranker-0.6B. The previous 60-document pools yielded 0.32409 and 0.32090 for the same policy, respectively. In held-out selection all folds choose policies operating on the original 30 or fewer candidates. The measured transfer reinforces the distinction between finding useful documents and confidently promoting them into a strong top ten. [8B ranking controls](../results/analysis/generated_temporal_view_8b_holdout_20261125.json) · [Yuanbao ranking controls](../results/analysis/generated_temporal_view_yuanbao_holdout_20261125.json) · [Raw scores](../results/analysis/generated_temporal_view_shortlist_scores_20261125.tar.gz)

The protected 60-document ranking from extractive retrieval scores 0.32448 with the frozen Yuanbao model and the full question. Score each document again with each extractive view and keep the same two excerpts selected from the full question. Standardize the per-query scores of each view, take the greater support for each candidate, then standardize the resulting vector. Blend 60% of the full-question score with 40% of this support score. The matched protected 60-document policy improves to **0.32752**, compared with the 0.32764 current reference. Fold-wise selection scores 0.31976 [−0.01800, +0.00204 in paired change]. The measured full-question and evidence-view inputs therefore leave the existing competition ranking in place. [Full-query ranking](../results/analysis/extractive_temporal_view_yuanbao_holdout_20261125.json) · [Three-view ranking](../results/analysis/extractive_temporal_view_combined_holdout_20261125.json) · [Per-view score cache](../results/analysis/extractive_temporal_view_combined_scores_20261125.tar.gz)

## Replay

Generate query embeddings from the [original 30-document training manifest](RERANKING_HEADROOM_20261125.md#reproduction). Use the 60-document training manifest from [the candidate-width study](QWEN_ADAPTATION_AND_PAIR_EVIDENCE_20261125.md#replay) for retrieval. Keep the corpus embeddings produced by the original dense stage.

```bash
python scripts/build_temporal_query_views.py --manifest "$work/train.json" \
  --data-root data/track1_tempo --rewriter models/Qwen3-4B-Instruct-2507 \
  --encoder models/mxbai-embed-large-v1 --extractive --output "$work/views.npz"

python scripts/retrieve_temporal_query_views.py --manifest "$work/wide-train.json" \
  --data-root data/track1_tempo --embeddings emb --runs runs \
  --vectors "$work/views.npz" --output "$work/view-pools.json" --audit "$work/view-oracles.json"
```

Remove `--extractive` to reproduce the generation control. Read `interleaved60` from the candidate list object and save it under `queries` for `score_specialist_shortlists.py`. Use the full original question for reranking. Run `eval_qwen_reranker_adapter.py` with the original 4B two-excerpt scores as the common matched control and the new pool's scores for each tested model.

For each of the two extractive scoring runs, add that query view as `scoring_query` in the candidate manifest while retaining the original `query` field. Score the original and the two view manifests with the same checkpoint, then pass the matched three score archives as `--full`, `--first`, and `--second` to `merge_temporal_view_scores.py`. Evaluate the merged scores and the original scores together.
