# Choosing complementary evidence at a fixed 8B budget

Use the current 0.38770 policy as the reference. A six-pair acceptance follows the same first-result protection and same-ten reorder. On an abstention, always keep the starting ten and choose 24 additional documents from the dense top 100. All alternatives keep the first 12 dense outsiders. With the remaining 12 slots, compare dense order, lexical coverage, incremental clause coverage, and teacher-score prediction.

The lexical features use the same 112-word window and tokenization as the original shared-reference prompts. Weight each query term by 1 plus the log ratio of 101 to one plus its document frequency within the 100 selected windows. The lexical arm selects documents by weighted query-token overlap. The clause arm splits the question at sentence boundaries and adds documents greedily by the sum of improvements over the best coverage achieved so far for each clause. Ties favor higher lexical coverage and earlier dense rank.

The student fits a ridge regression with regularization 0.1 to predict the normalized full-pool 8B scores of training documents. Its five inputs are frozen fusion, reciprocal dense rank, weighted lexical coverage, maximum clause coverage and mean clause coverage. Fit on one of the two held-out cohorts and evaluate the other, then fit on all 30 training abstentions and transfer to the 263 official dev abstentions. Combine the predicted judge score with the frozen score at 30:70 to order the final 12 candidates. An additional diagnostic chooses those 12 from the cached full-pool 8B fusion. The latter selector uses scores for all 100 documents and estimates a teacher-based alternative, with a larger selection budget.

After any selector chooses a 34-document pool, recompute the same local 70:30 fusion and score the resulting top ten. All effects refer to the complete policy over 65 held-out or 519 dev queries with equally weighted domains. The deployable arms keep the current 54.43 estimated prompts per dev query. [8B budget and calibration](BUDGETED_ANCHOR_FALLBACK_20261125.md) · [Accepted-list rule](ACCEPTED_WEIGHT_FULL_DEV_20261125.md)

## Fixed-budget outcomes

| Selector for the remaining 12 candidates | Training holdouts | Official dev | Change in dev label-based candidate ceiling | Mean replaced candidates on dev abstentions |
|---|---:|---:|---:|---:|
| Continue in dense order | 0.31102 | **0.38770** | 0 | 0 |
| Lexical coverage | 0.31302 | 0.38685 | −0.00294 | 9.65 |
| Complementary clause coverage | **0.31386** | 0.38674 | −0.00152 | 9.71 |
| Student prediction | 0.30664 | 0.38674 | **+0.01004** | 6.17 |
| Full 8B teacher selection, diagnostic | 0.31002 | 0.38546 | **+0.02554** | 6.87 |

Complementary clause coverage is the deployable selector chosen by out-of-fold training scores. Its dev change is −0.00097 [−0.00384, +0.00103]. The candidate ceiling places all judged positives available in the pool at the earliest ranks on covered queries. Replace only the abstention candidate pools when aggregating its change. The student and full-score diagnostic increase that ceiling and yield lower realized point estimates under local fusion. [Candidate selection and realized rankings](../results/analysis/evidence_shortlist_20261125.json)

## Common normalization isolates the candidate selection effect

Keep the dense and full-teacher candidate pools fixed across normalization arms. Both contain the same 22-document core. Standardize the two score sources over the selected 34, the shared core, or the full top 100 as shown in the table. Within each of the three common-scope arms, the score of a shared document is identical under the two candidate selectors.

| Frozen-score scale | 8B score scale | Dense selector on dev | Full-teacher selector on dev | Full teacher minus dense |
|---|---|---:|---:|---:|
| Individual 34 | Individual 34 | 0.38770 | 0.38546 | −0.00224 |
| Common 22 | Common 22 | 0.38623 | 0.38630 | +0.00007 |
| Full 100 | Common 22 | 0.38541 | 0.38578 | +0.00037 |
| Full 100 | Full 100 | 0.38691 | 0.38713 | +0.00022 |

Under separate 34-document scaling, the teacher selection changes the order on 151 abstentions, including 71 with changed membership. Under full common scaling, all 62 changed rankings also change membership. Reusing the same scale removes the point-estimate penalty of teacher selection, and the matched gains remain small. For full frozen scaling with the common 22 judge scores, the +0.00037 difference has paired interval [−0.00004, +0.00098]. The absolute results show the remaining gap between retrieving judged positives into a shortlist and giving them high enough scores to place in its first ten. [Matched scale and membership replays](../results/analysis/teacher_shortlist_normalization_20261125.json)

## Reproduction

```bash
python scripts/replay_evidence_shortlist.py --data-root data/track1_tempo \
  --budgeted results/analysis/budgeted_anchor_fallback_20261125.json \
  --protected results/analysis/accepted_weight_full_dev_audit_20261125.json \
  --oof results/analysis/anchor_reranker_qwen8b_sample26_20261125.json results/analysis/anchor_reranker_qwen8b_replication39_20261125.json \
  --dev results/analysis/anchor_reranker_qwen8b_dev100_20261125.json \
  --oof-scope results/analysis/anchor_edit_scope_qwen8b_oof100_20261125.json \
  --dev-scope results/analysis/anchor_edit_scope_qwen8b_dev100_20261125.json \
  --oof-frozen results/analysis/anchor_edit_scores_oof100_20261125.json \
  --dev-frozen results/analysis/anchor_edit_scores_dev100_20261125.json \
  --output results/analysis/evidence_shortlist_replay.json
```

For the matched normalization replay, use the same arguments, add `--diagnose-normalization`, and provide a fresh output filename.
