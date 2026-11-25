# Grade individual documents after checking their evidence

Prior score-source tests repeatedly improved standalone ranking without transferring much of that gain to the final list. This test gives the same model either no intermediate evidence step or a separate step to quote and interpret the displayed text. The initial ranking and the 14 candidate IDs are shared by all controls. [Matched reranker studies](QWEN_ADAPTATION_AND_PAIR_EVIDENCE_20261125.md) · [Candidate-width and query-view controls](TEMPORAL_QUERY_VIEWS_20261125.md)

## Fixed candidate pools and matched ratings

Keep the current top ten and add four documents. Choose two outsiders from the first 30 candidates and two from the additional 30. Within each tier, average the standardized two-excerpt scores from Qwen3-Reranker-8B and Yuanbao-Reranker-0.6B to choose the two outsiders. This retains the full current ranking and exposes otherwise lower-ranked evidence for judgment. The labeled ideal score of the 14-document pool is **0.55238** on the 186 training holdouts. At least one of the four extra candidates is judged relevant on 33 queries. Ten of those queries have no judged positive in their current top ten.

Use Qwen3-4B-Instruct-2507 with the full question and the same two excerpts. It grades candidates on an ordinal scale from 0 for irrelevant or conflicting to 3 for a clear answer to a specific requirement. Direct scores use the model's probabilities of the four rating tokens without a generated explanation. The evidence-first arm generates a short quote and reasons about the requirement it supports before predicting the rating tokens. The verified arm uses that evidence-first rating when at least five consecutive normalized words in the quotation occur in the displayed excerpts. It uses the direct rating for the remaining judgments. The decoder produces 1,516 verified quotations across 2,604 document assessments. Verification establishes that the quotation came from the provided text.

Ranking policies use the expected ordinal rating and combine its standardized value with the standardised rank discount for the preceding list. Compare 30% and 50% judge weights, with optional rank-one protection. Add a single-edit policy as a control. It can swap the first document with ranks two through five or admit one challenger at ranks eight through ten. The minimum expected-grade difference is 0.65 and the challenger grade must reach 1.8.

## Training-holdout results

| Policy on 186 held-out queries | Domain-macro nDCG@10 |
| --- | ---: |
| Current reference | 0.32764 |
| Best pooled direct-rating policy | 0.33301 |
| Direct rating, 50% blend | 0.33083 |
| Evidence-first rating, 50% blend | 0.33512 |
| Verified quotation with direct fallback, 50% blend | **0.33638** |
| Policy chosen on the other two folds | **0.33266** |

The cross-fold increase is **+0.00502** [−0.00469, +0.01509]. On the same 65 exact-reference queries, the cross-fold increase is +0.00676. The smallest leave-one-domain-out gain is +0.00205. The selected fold-wise policies add 16 judged relevant documents and lose nine. Ten of the additions come from the second 30 candidates. Four zero-score queries gain a positive and two become new zeroes. The official evaluation applies the pooled choice to all 519 queries with the same candidate selection, excerpt selection, checkpoint and thresholds. [Held-out rankings and matched controls](../results/analysis/evidence_reasoner_holdout_20261125.json) · [Archived ratings and verification indicators](../results/analysis/evidence_reasoner_holdout_scores_20261125.tar.gz)

On the 165 holdout queries with step judgments, the cross-fold policies raise domain-macro coverage from **60.04% to 62.33%**. They cover eight previously missing steps and lose six. In history query `62622_605`, the system adds a document about archival personnel cards that can contain salary records. In bitcoin query `126019_1`, it admits a documented observation about one-block reorganizations when the request concerns reorganizations of two to four blocks. The latter illustrates how a traceable quotation can still refer to the wrong scope.

## Widen the evidence judgment pool

Keep the original fourteen candidates and cached ratings. Score four more outsiders from each retrieval tier using the same grader. Expanding to 22 documents raises the holdout candidate ceiling from 0.55238 to 0.62708 and increases the number of queries with a reachable positive despite a zero-score reference from 10 to 21.

With the same fixed verified 50% policy, expanding the pool raises nDCG@10 from 0.33638 to **0.33797**, a paired increase of +0.00159 [−0.00705, +0.01020]. The increment on the 65 exact-reference queries is +0.00352. Fixing normalization to the common original fourteen yields 0.33758 with the larger pool. Choosing among all controls across held-out folds yields 0.33069. The fixed-rule comparison supports a wider-pool follow-up when the original official evaluation shows improvement below the target. [22-document holdout controls](../results/analysis/evidence_reasoner_width22_holdout_20261125.json) · [Ratings for the eight additional candidates](../results/analysis/evidence_reasoner_width22_holdout_scores_20261125.tar.gz)

## Official dev with the rule fixed on training

Score 7,266 document candidates for all 519 official dev queries. The 14-document pool has a judged ceiling of 0.59576. The matched judge controls use the same 50% rank blend.

| Method | Domain-macro nDCG@10 |
| --- | ---: |
| Current reference | 0.39589 |
| Direct rating | 0.38815 |
| Quote and explanation before rating | **0.40059** |
| Verified quotation with direct fallback, training-selected | 0.39536 |

The training-selected policy changes nDCG@10 by −0.00053 [−0.00980, +0.00782]. The smallest leave-one-domain-out change is −0.00318. Step coverage rises from **65.65% to 67.86%**. The policy adds 48 judged positives and removes 45, recovers ten zero-score queries and creates six, and changes the first document on 41 queries. The per-domain result is positive in five domains, with the largest gain in Economics (+0.03122) and the largest decline in Law (−0.04112).

The explanation raises nDCG@10 by **+0.01244** over the same model grading directly. This matched difference isolates the intermediate generation step. Exact quotation matching accepts 3,856 of the 7,266 generations. Applying the training-selected fallback to direct grading for the other 3,410 costs 0.00523 compared with grading from all explanations. The 22-document evaluation queue uses the earlier requirement of a positive gain from the locked 14-document policy. [Official rankings](../results/analysis/evidence_reasoner_official_dev_20261126.json) · [Rating and verification cache](../results/analysis/evidence_reasoner_official_scores_20261126.tar.gz)

Fit domain-weighted relevance calibration on the exact same 14 candidates. Match candidate-position features across the pointwise-only, ordinal-evidence-only, and combined feature arms. Out of fold the three fitted arms score 0.32543, 0.33152 and 0.33263, and all folds favor the fixed verified blend at 0.33638. [Calibrated holdout replay](../results/analysis/calibrated_evidence_reasoner_holdout_20261126.json)

## Combine rankings across evidence views

Combine four lists with equal votes using reciprocal rank. The inputs are the current reference, the training-selected 8B pointwise blend, and the original and verified 50% evidence blends. Each list contributes `1/(c+r)` at one-based rank `r`, with a zero contribution outside the first ten. Select among the eight fixed arms on the 186 training holdouts. The four-list rule at `c=5` scores 0.33759 when applied to the pooled holdouts. Selection across folds scores 0.33435, up 0.00670 from the holdout reference. [Reciprocal rank fusion](https://doi.org/10.1145/1571941.1572114)

| Fixed ranking on all 519 official queries | Domain-macro nDCG@10 | Change from reference |
| --- | ---: | ---: |
| Current reference | 0.39589 | — |
| Matched 8B pointwise ranking | 0.39893 | +0.00304 |
| Same-checkpoint explanation before rating | 0.40059 | +0.00469 |
| Equal four-list votes, (c=5) | 0.40392 | **+0.00802** |
| Equal four-list votes, (c=60) | **0.40400** | **+0.00811** |

The original four-list rule gains +0.00802 with a paired 95% interval [+0.00428, +0.01208] and a domain-resampling interval [+0.00487, +0.01111]. Every leave-one-domain-out gain is at least +0.00730. The extension compares three more rank offsets, the additional 60-candidate 8B list, and a listwise Jina vote. The pooled holdout choice keeps the same four sources at (c=60). It gains +0.00811 [+0.00426, +0.01224] on official dev, with at least +0.00739 after omitting any domain. Both policies improve 11 of 13 domain means. The (c=60) policy adds 29 judged positives and removes 19. It rescues seven zero-score queries, creates two and changes the first document only 12 times. New and removed documents together contribute +0.00448. The remaining +0.00363 comes from reordering retained documents. Step coverage moves from 65.65% to 66.99%.

The two evidence variants share a checkpoint and cast separate votes. Their common improvement over direct grading motivates testing both signals alongside the externally reranked list. The 22-document control with the same four sources scores 0.33724 on the training holdouts compared with 0.33759 for 14. Mean position and two pairwise-majority aggregation rules retain the same pooled policy choice and reduce cross-fold performance. [Ranking consensus and matched controls](../results/analysis/evidence_ranking_consensus_20261126.tar.gz)

## Let two scorers justify a change at rank one

Start from the training-selected four-list consensus at offset 60. A candidate in positions two through five can swap into the first position if it also appears in the 8B pointwise top five. The candidate's evidence grade must reach 2 on the 0–3 scale. Choose the strongest eligible candidate by its grade and require a minimum gap over the incumbent. The pooled training holdout selects a 0.5 gap and requires a quotation found in the displayed evidence. This preserves the ten document IDs and makes at most one swap per query.

| Official dev, 519 queries in 13 domains | Domain-macro nDCG@10 |
| --- | ---: |
| Current reference | 0.39589 |
| Four-list rank consensus | 0.40400 |
| Rank consensus with the selected first-position rule | **0.40674** |

The full policy gains **+0.01085** [ +0.00266, +0.01920 ] over the reference. The domain-resampling interval is [+0.00430, +0.01775]. Eleven domain means improve and the smallest leave-one-domain-out gain is +0.00878. The first-position rule changes 95 queries relative to the consensus stage. Of these, 16 score higher, 22 score lower and 57 tie under the qrels. The extra gain over consensus is **+0.00274** [−0.00408, +0.00976]. On the training holdouts the same fixed rule gains +0.00014 over consensus, while selecting the first-position rule across folds scores 0.33261 against 0.33455 for cross-fold consensus selection. Those comparisons support using the consensus effect as the primary result and reporting the first-position rule as a subsequent dev improvement.

The ranking retains the consensus coverage of 66.99% and admits the same 29 judged positives while dropping 19 relative to the original reference. New and removed documents contribute +0.00448 overall. Ordering the retained documents contributes +0.00637. This distinguishes agreement on which evidence belongs in the ten from agreement on which evidence deserves first position. [Training selection](../results/analysis/evidence_consensus_head_holdout_20261126.json) · [Official rankings and statistics](../results/analysis/evidence_consensus_head_official_dev_20261126.json)

## Matched controls for reading and grading

Replay the same 186-query, 14-document population. Mask explicit numerical grades in the generated notes before continuing the original rating prompt. The fixed verified blend moves from 0.33638 to 0.33636. Cross-fold selection reaches 0.33358. A fresh grader that receives the document and optionally the same extracted notes reaches 0.33131 with its best fixed blend and 0.32409 with fold-wise selection.

Keep the first lexical excerpt and select the second for the highest coverage of new terms within one question clause. The direct 50% blend rises from 0.33083 to 0.33685 on the fixed holdout pool. Explaining those clause-targeted excerpts before rating scores 0.33133 under the same blend. Selecting among the controls by fold scores 0.33366. Input selection and intermediate reasoning interact even when the candidate IDs, rating prompt and model stay fixed. [Rankings and cached grades for all three controls](../results/analysis/evidence_reasoner_ablation_holdouts_20261126.tar.gz)

## Replay

Build a 60-document manifest and keep the matching Qwen3-Reranker-8B and Yuanbao scores from the [same-width study](QWEN_ADAPTATION_AND_PAIR_EVIDENCE_20261125.md#replay). Use the original reference repair report when building the 14-document candidate pools.

```bash
python scripts/eval_evidence_reasoner.py prepare --manifest "$work/wide-train.json" \
  --first "$work/8b-scores"/*.json --second "$work/yuanbao-scores"/*.json \
  --repair results/analysis/evidence_document_repair_20261125.json \
  --output "$work/reasoner-train.json"

python scripts/eval_evidence_reasoner.py score --manifest "$work/reasoner-train.json" \
  --data-root data/track1_tempo --model models/Qwen3-4B-Instruct-2507 \
  --batch-size 8 --max-new-tokens 80 --checkpoint "$work/reasoner-checkpoint" \
  --output "$work/reasoner-scores.json"

python scripts/eval_evidence_reasoner.py evaluate --manifest "$work/reasoner-train.json" \
  --data-root data/track1_tempo --scores "$work/reasoner-scores.json" \
  --output "$work/reasoner-selection.json"
```

For official dev, use the 519-query candidate manifest and pass the training report as `--selection` to the evaluator. The published cache retains ordinal scores and verification indicators. The raw generated evidence text remains in the experiment workspace.

For the 8B candidate scorer, `score_specialist_shortlists.py --gpu-memory-gib 8 9` allows two visible devices to share the BF16 weights with Accelerate. A 60-document replay across the two placements reproduces every logit difference exactly.

Use `prepare --per-tier 6` to build the 22-document pools. The first fourteen IDs retain their original order. Run `score` on the final eight IDs, then pass both the original and additional score archives to `evaluate` for the complete 22-document manifest.

Run `replay_evidence_ranking_consensus.py` on the evidence-reasoner and 8B reports for each split. Supply the Jina report as `--listwise` to reproduce the offset and added-voter controls. For the first-position experiment, pass the resulting consensus report to `replay_consensus_head.py --reference` and the evidence-score JSON shards as `--evidence`. Both scripts take the original qrels directory as `--data-root`. Use the corresponding training output as `--selection` on official dev.

Use `rescore_evidence_ratings.py` and `grade_evidence_independently.py` with fresh unredacted local reasoner output for the two grading controls. Use `eval_evidence_reasoner.py score --evidence-mode clause` for the clause-targeted evidence control. The published grades can be evaluated directly with the same held-out manifest.
