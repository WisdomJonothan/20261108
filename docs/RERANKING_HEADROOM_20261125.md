# Where the current RETECO ranking can gain more

The reference is the 0.39269 directional-passage policy on all 519 public dev queries. All experiments retain the 13-domain macro aggregation. This public dev set has been examined in earlier rounds. [Current reference and cached scores](CANDIDATE_ADMISSION_AND_CONTAINMENT_20261125.md)

## Locate the larger gaps

This audit uses official qrels to calculate ideal rankings under explicit candidate and position constraints. A score in the table is a diagnostic ceiling except for the measured reference. The shortlist consists of the current ten, ten additional documents ordered by cached 70:30 frozen/8B fusion within dense depth 100, and ten more documents from mxbai ranks 101–110.

| Allowed evidence and ordering | Official dev nDCG@10 | Queries with no judged positive in the list or pool |
|---|---:|---:|
| Current system | 0.39269 | 121 |
| Ideal reorder of current ten, keeping rank one | 0.45480 | 121 |
| Ideal reorder of current ten, all positions open | 0.53917 | 121 |
| Ideal order from the first 20 study candidates | 0.61548 | 87 |
| Ideal order from all 30 study candidates | 0.63230 | 83 |
| Ideal order from mxbai top 100 | 0.74462 | 52 |
| Ideal order from mxbai top 300 | 0.85132 | 26 |

Releasing rank one adds 0.08437 to the same-ten ceiling. It changes the ceiling for 224 queries. Widening to 20 documents makes the first labeled positive reachable on 34 current zero-score queries. These estimates motivate two distinct interventions, better ranking near the front and recovery from a different candidate pool. [Per-query headroom audit](../results/analysis/rerank_headroom_audit_20261125.json)

## Match the evidence input and the scoring model

Keep one Qwen3-Reranker-4B checkpoint with the published yes/no scoring protocol. Compare the general retrieval instruction, an instruction describing temporal evidence for the question, and the temporal instruction with a second excerpt selected for query terms missing from the first excerpt. The first excerpt is always the highest lexical-match 112-word window. The tasks share the full question and candidate IDs. The temporal instruction treats evidence for one part of a question as useful when it respects the relevant time restriction. [RETECO track definition](https://datascienceuibk.github.io/RETECO/task.html) · [Qwen3-Reranker-4B protocol](https://huggingface.co/Qwen/Qwen3-Reranker-4B)

Test 186 training-fold queries. The 65 earlier held-out queries use the actual 8B reference. The other 121 are selected by a fixed domain-stratified hash, use the out-of-fold six-pair reference, and fill the near portion of the candidate pool from dense rank. For all queries, the shared ranking policy takes a z-standardized reference rank discount and a z-standardized yes-minus-no logit, calculated inside the chosen candidate list. The locked policy uses 20 candidates and 30% temporal judge weight. Its matched generic-prompt control and the extra-excerpt condition use the same 4B checkpoint.

On the matched 20-candidate, 30%-judge policy, training-fold scores are 0.32102 with the general prompt and 0.32542 with the temporal prompt. The single-passage selection gains +0.00464 [−0.00517, +0.01443] on cross-fitted 186-query evaluation. On the 65 actual-reference queries, the pooled temporal policy rises from 0.30993 to 0.32136. Adding the two-excerpt arm yields a training-selected 30-candidate policy with 50% judge weight and rank-one protection. The blocked three-fold gain is +0.00983 [−0.00144, +0.02128]. On the 65 actual-reference queries, the pooled two-excerpt policy scores 0.32905. Both official-dev configurations were fixed using training scores before their corresponding official scoring runs. [Single-passage training controls](../results/analysis/rerank_headroom_training_prompt_comparison_20261125.json) · [Two-excerpt training controls](../results/analysis/rerank_headroom_training_context_comparison_20261125.json) · [Cached training logits](../results/analysis/rerank_headroom_training_scores_20261125.tar.gz)

## What the single-passage reranker changes on official dev

The temporal instruction with 20 candidates scores **0.39462**, a gain of **+0.00193** [−0.00524, +0.00929]. It preserves rank one on every query at this weight. It admits 58 judged relevant documents and removes 54 from the top ten. In the exact discounted attribution, admitted gold documents contribute +0.01371, removed gold documents contribute −0.01221, and reordering retained documents contributes +0.00044. Seventeen reference-zero queries recover a positive while twelve become new zeroes. Annotated-step coverage on the 484 annotated queries rises from 63.12% to 65.14%. A 20-document deployment adds 20 pointwise 4B prompts per query. The evaluation scores all 30 documents for the matched candidate-width controls. [Official dev ranking and attribution](../results/analysis/rerank_headroom_temporal_dev_20261125.json) · [Raw official-dev temporal logits](../results/analysis/rerank_headroom_temporal_scores_20261125.tar.gz)

## Two excerpts and confirmation of individual admissions

The training-selected two-excerpt policy uses the 30-candidate pool, a 50% 4B judge weight, and rank-one protection. It scores **0.39488**, an increase of **+0.00219** [−0.00788, +0.01241] over the reference. Its step coverage is **65.25%**. It admits 68 judged relevant documents and removes 63. The exact contributions are +0.02010 from added documents, −0.01595 from removed documents, and −0.00195 from repositioning retained documents. The policy adds 30 4B scores per query. Relative to the selected one-excerpt policy, the point estimate increases by 0.00026. [Full dev rankings and attribution](../results/analysis/rerank_headroom_dispersed_dev_20261125.json) · [Raw two-excerpt logits](../results/analysis/rerank_headroom_dispersed_scores_20261125.tar.gz)

Confirm individual admissions with the other evidence view by pairing entering and displaced documents in order of their scores in that view. Restore a displaced document when the corresponding proposed entrant has a lower score or ties. The temporal list followed by this confirmation reaches **0.39507** on dev [+0.00238, paired 95% interval −0.00479 to +0.00965 versus the reference]. Its training-fold result is 0.32449 compared with 0.32542 before confirmation. The training selection retains the two-excerpt policy at 0.39488 on dev. Both views use the same checkpoint and differ in the excerpt selection. [Confirmation in training folds](../results/analysis/rerank_headroom_admission_confirmation_train_20261125.json) · [Official dev controls](../results/analysis/rerank_headroom_admission_confirmation_dev_20261125.json)

## Measure candidate-source complementarity

From the original mxbai top 100, add the highest 24 BGE v1.5 candidates outside that set. The dev oracle increases by +0.02690 to 0.77153. Seven queries that have no judged positive in mxbai top 100 reach one in the union. Regenerate training-side BGE retrieval from the cached document vectors. Fit logistic regression and histogram-based gradient boosting on 969 training-fold queries using score and rank features from mxbai, BGE and the out-of-fold frozen fusion. Weights give each represented training domain equal influence. Nested held-out folds choose the applied ranking policy. The training-selected pooled linear fit on the union scores 0.38700 on dev, a change of −0.00569 [−0.01114, −0.00103]. It adds 21 judged relevant documents and removes 36 from the top ten. This experiment separates the presence of new relevant candidates from the ability of the scoring model to rank them. [All cross-retriever controls and rankings](../results/analysis/cross_retriever_ranker_20261125.json.gz)

## Replaying confirmation after directional deduplication

Apply the prior two-judge rule with exactly the same eligibility and comparison thresholds after the directional-passage intervention. It can replace rank 9 or 10 once. The resulting exploratory dev score is **0.39353**, an increment of +0.00083 [−0.00126, +0.00328] over the reference, with 15.20 extra 4B scores per query. The same fixed rule adds +0.00284 on the 65 training holdouts. Pooled training selection across all tail-edit arms still favors the 8B-only wider rule, which scores 0.39088 on dev. [Composed policy and matched controls](../results/analysis/corroborated_directional_admission_20261125.json)

## Reproduction

Use the archived candidate runs and the official data directory. The commands below reuse cached scores. For fresh inference on a manifest, run `score_specialist_shortlists.py` with the same 4B checkpoint, `--batch-size 16` and the specified `--evidence-mode`.

```bash
work=/tmp/reteco-headroom-replay
mkdir -p "$work/train-scores" "$work/temporal-scores" "$work/dispersed-scores"
python scripts/build_rerank_headroom_study.py --data-root data/track1_tempo --runs-root runs \
  --manifest "$work/manifest.json" --audit "$work/audit.json"
python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); q=json.loads((p/"manifest.json").read_text())["queries"]; [(p/f"{s}.json").write_text(json.dumps({"queries":[r for r in q if r["split"]==s]})) for s in ("train","dev")]' "$work"

tar -xzf results/analysis/rerank_headroom_training_scores_20261125.tar.gz -C "$work/train-scores"
tar -xzf results/analysis/rerank_headroom_temporal_scores_20261125.tar.gz -C "$work/temporal-scores"
tar -xzf results/analysis/rerank_headroom_dispersed_scores_20261125.tar.gz -C "$work/dispersed-scores"

python scripts/eval_rerank_headroom_study.py --data-root data/track1_tempo --manifest "$work/train.json" \
  --scores "$work/train-scores"/score-train-{reference,temporal}.json --output "$work/selection-two.json"
python scripts/eval_rerank_headroom_study.py --data-root data/track1_tempo --manifest "$work/train.json" \
  --scores "$work/train-scores"/score-train-{reference,temporal,dispersed}.json --output "$work/selection-three.json"
python scripts/eval_rerank_headroom_study.py --data-root data/track1_tempo --manifest "$work/dev.json" \
  --scores "$work/temporal-scores"/*.json --selection "$work/selection-two.json" --output "$work/temporal-dev.json"
python scripts/eval_rerank_headroom_study.py --data-root data/track1_tempo --manifest "$work/dev.json" \
  --scores "$work/dispersed-scores"/*.json --selection "$work/selection-three.json" --output "$work/dispersed-dev.json"
python scripts/replay_two_view_admission.py --data-root data/track1_tempo --manifest "$work/train.json" \
  --scores "$work/train-scores"/score-train-{temporal,dispersed}.json \
  --output "$work/confirmation-train.json"
python scripts/replay_two_view_admission.py --data-root data/track1_tempo --manifest "$work/dev.json" \
  --scores "$work/temporal-scores"/*.json "$work/dispersed-scores"/*.json \
  --selection "$work/confirmation-train.json" --output "$work/confirmation-dev.json"
```

To reproduce the cross-retriever ranker, generate the BGE v1.5 training runs with `run_dense_bge.py --split train --topk 300` in each domain using the original embedding cache. Pass the training and existing dev BGE directories to `replay_cross_retriever_ranker.py`. For the tail confirmation control, pass `--directional results/analysis/directional_passage_containment_20261125.json` to `replay_corroborated_tail_admission.py`.
