# Candidate admission and directional passage overlap

The fixed-budget training proxy selects directional passage penalties. That rule moves official dev domain-macro nDCG@10 from **0.39227 to 0.39269** across all 519 queries. A separate, higher-budget diagnostic for two-model agreement reaches **0.39338**. The official dev set was inspected in earlier experiments. All results use the same 256 accepted lists and evaluate changes on the 263 six-pair abstentions. [Starting score and candidate protocol](CANDIDATE_WIDTH_AND_DUPLICATE_SEVERITY_20261125.md)

## Directional overlap at the original prompt budget

The reference penalizes symmetric Jaccard overlap on the five-token sequences of the query-matched passage. A shorter passage can repeat a longer one almost entirely while getting a low symmetric similarity score because the longer passage has additional text. Measure the fraction of the candidate's sequences that also occur in an already selected passage. The candidate must have at least 12 distinct sequences and between 35% and 95% of the retained passage's sequence count. The 95% length bound is exclusive. Apply the larger of the original penalty and the new penalty at each greedy selection step. The new penalty is either the measured fraction or twice that fraction, subject to a minimum containment threshold. A shorter retained passage adds no directional penalty to the longer candidate.

| Additional directional penalty | 523-query training proxy | 65-query 8B holdout | 519-query official dev |
|---|---:|---:|---:|
| Reference | 0.277630 | 0.30993 | 0.39227 |
| Fraction when at least 90% contained | 0.277986 | 0.30993 | 0.39253 |
| Fraction when at least 95% contained | 0.277980 | 0.30993 | 0.39254 |
| Twice fraction when at least 90% contained | **0.277998** | 0.30993 | **0.39269** |
| Twice fraction when at least 95% contained | 0.277991 | 0.30993 | 0.39289 |

The pooled training proxy chooses the doubled 90% rule. Holding out each of the three training folds selects the doubled 90% rule twice and the single 90% rule once. The held-out pooled proxy gain is +0.00036 [−0.00055, +0.00137]. All variants tie on the 65 queries with actual 8B scores. The selected rule increases official dev by **+0.00042** [−0.00039, +0.00146]. Five queries increase their nDCG and one decreases. Twenty-six top-ten rankings change, with a membership change on 25. Travel provides the largest domain contribution. Annotated-step coverage averages 63.12%, compared with 63.07% before the change. The rule retains the 34-candidate pool, preserves rank one, and uses the original 54.43 estimated prompts per dev query. [Per-query containment rankings and paired results](../results/analysis/directional_passage_containment_20261125.json)

## Matched controls for one replacement

Keep the 34 candidates and the current top ten. A proposed replacement can score at most 0.35 blended-score units below the removed document and must have Jaccard similarity below 0.8 to every retained passage. For the last-two-rank comparison, allow at most one edit at rank 9 or 10. Compare approval by the cached Qwen3-8B judge, approval by the Qwen3-Reranker-4B specialist, and approval by both. In the joint rule, choose the swap maximizing the smaller of the two standardized margins. The rank-6–10 and narrower score-margin variants are in the linked artifact.

| Last-two-rank policy | 65-query 8B holdout | 519-query official dev | Extra prompts per dev query |
|---|---:|---:|---:|
| Keep reference | 0.30993 | 0.39227 | 0.00 |
| Require 8B approval | 0.31483 | 0.39066 | 0.00 |
| Require specialist approval | 0.31277 | 0.39169 | 16.97 |
| Require both approvals | 0.31277 | **0.39338** | 14.87 |

The extra prompt count allows the specialist to skip queries without a qualifying pair, then counts all 34 scores needed to standardize the specialist margin on each eligible query. The agreement rule raises official dev by +0.00111 [−0.00113, +0.00364] relative to the reference. It improves eight queries and lowers ten. Relative to the single 8B last-two-rank control, its paired gain is +0.00272 [+0.00026, +0.00584]. Relative to the specialist control, it adds +0.00169 [−0.00010, +0.00398]. These intervals resample queries within domains. The pooled 65-query selection across all arms favors the 8B-only rank-6–10 rule, which scores 0.39075 on dev. Cross-fitting between the two training cohorts yields the reference score. Treat the agreement result as a higher-budget diagnostic on previously inspected dev. [Full score and edit audit](../results/analysis/corroborated_tail_admission_20261125.json)

## Preserve the original normalization when expanding candidates

For 58 or 100 candidates, compare local normalization with scaling both the frozen and 8B scores on the original 34 documents. The one-admission variant additionally protects the reference first result and allows at most one document from outside the 34 into the final ten. Use the reference symmetric duplicate penalty throughout.

| Candidate width and rule | 65-query 8B holdout | 519-query official dev | Extra prompts per dev query |
|---|---:|---:|---:|
| 34 reference | 0.30993 | 0.392270 | 0.00 |
| 58, local scaling | 0.31177 | 0.390672 | 24.32 |
| 58, 34-document scaling | 0.31166 | 0.391436 | 24.32 |
| 58, 34-document scaling and one admission | 0.31166 | 0.392268 | 24.32 |
| 100, local scaling | **0.31538** | 0.390690 | 66.89 |
| 100, 34-document scaling | 0.31166 | 0.391611 | 66.89 |
| 100, 34-document scaling and one admission | 0.31166 | 0.392421 | 66.89 |

The 523-query proxy selects the 58-document one-admission policy, and the 65-query holdout selects local scaling at 100. Under both widths, carrying the small-pool scale into the bigger pool recovers part of the drop from local normalization. Protecting the first result and capping admissions recovers more of that drop. [Expansion rankings and comparisons](../results/analysis/shortlist_calibrated_expansion_20261125.json)

## Reproduction

Use the archived judge scores, the official corpus and qrels, and fresh output paths. All three commands replay on CPU.

```bash
python scripts/replay_shortlist_calibrated_expansion.py --containment \
  --data-root data/track1_tempo --output results/analysis/directional_passage_containment_replay.json

python scripts/replay_shortlist_calibrated_expansion.py \
  --data-root data/track1_tempo --output results/analysis/shortlist_calibrated_expansion_replay.json

python scripts/replay_corroborated_tail_admission.py \
  --data-root data/track1_tempo --output results/analysis/corroborated_tail_admission_replay.json
```
