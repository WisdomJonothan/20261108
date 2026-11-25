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
