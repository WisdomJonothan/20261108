# Which event owns the date?

The [earlier RETECO contrast](TEMPORAL_QUERY_CONTRAST_20261125.md) changed the time condition against fixed real passages. This follow-up asks whether rankers can associate a date with the requested event when the two documents contain the same words and years. [MenatQA](https://aclanthology.org/2023.findings-emnlp.100/) motivates temporal sensitivity tests in question answering. Here the outcome is the preference between two documents.

## Protocol fixed before scoring

- Cross six fictional site-event topics with six strictly ordered year triples to create 36 families. Each pair assigns the same two years to the same two named sites in opposite ways. Check identical word multisets for the intact passages. Repeat each family in both sentence orders. For each order cross both requested sites with both relations, before and after, for 288 document-pair judgments per arm.
- Score each query and each document independently. Keep the scoring function fixed and compare both dates visible, only the other site's date hidden, and only the requested site's date hidden. Both masking arms use the same placeholder. Strict success requires all four judgments for a sentence order to be correct, with ties counting as failures. Family success requires both sentence orders. In the target-hidden control, matching the original label measures preference retention after its supporting year has been removed.
- Reuse the 11 exact official-training source pairs with the committed event-date annotations. Compare the pristine text against the same text with an explicitly unrelated date note. The dates in that note either follow or exchange the two passages' annotated times. The note leaves the gold choice unchanged. A strict flip requires both temporal query variants to prefer their expected passage. These purposively selected real pairs serve as a sensitivity check.
- Run the local Qwen3-Reranker-0.6B with the default relevance instruction and the base BGE-reranker-v2-m3 cross-encoder in BF16. Preserve raw scores and the deterministic cases. This is a diagnostic on synthetic text and training-corpus evidence, with no calibration or official dev or test scoring.

## Matched date assignment

| Scorer and visible evidence | Original correct preferences / 288 | Ties / 288 | All four correct / 72 orders | Both orders correct / 36 families |
| --- | ---: | ---: | ---: | ---: |
| Qwen 0.6B, both dates | 252 | 17 | 42 | 16 |
| Qwen 0.6B, other date hidden | 268 | 6 | 55 | 24 |
| Qwen 0.6B, target date hidden | 50 | 16 | 0 | 0 |
| Base BGE, both dates | 183 | 0 | 0 | 0 |
| Base BGE, other date hidden | 246 | 3 | 41 | 18 |
| Base BGE, target date hidden | 32 | 0 | 0 | 0 |

Masking the other date makes 11 new Qwen families succeed and costs three previous successes. Base BGE gains 18 families. When only the other date remains, Qwen selects the side that matches the query's temporal relation for the wrong event in 222 judgments, and BGE does so in 256. Together the two masking directions indicate that temporal comparison is available and can be applied to the wrong event. The strength of the effect depends on the model and the context. The year triple and the site-event wording are reused across families, so these counts describe this controlled grid.

## Verified source passages

| Scorer | Pristine variants / 22 and flips / 11 | Aligned note | Exchanged note |
| --- | ---: | ---: | ---: |
| Qwen 0.6B | 13 and 2 | 12 and 1 | 11 and 0 |
| Base BGE | 13 and 2 | 13 and 2 | 12 and 1 |

Exchanging explicitly unrelated dates reduces full flips for both scorers relative to the aligned note. The pristine texts already yield only two flips per scorer. These 11 fixed cases suggest sensitivity to irrelevant dates in authentic evidence and give a concrete follow-up: test whether ranks improve when the correct event and its date can be extracted together, with an equally short control and a fallback when the link is ambiguous. The masking treatment in the controlled grid uses the template's known event boundaries. A retrieval method needs a separate test of its own event identification on held-out natural queries.

## Replay

Run the same command once per scorer with its local checkpoint. Use `--kind bge` and a fresh output path for the base BGE comparison.

```text
python scripts/eval_temporal_date_binding.py --kind qwen --model <local-qwen3-reranker-0.6b> --device cuda --precision bfloat16 --data-root <track1-tempo> --manifest results/analysis/temporal_minimal_pairs_manifest_20261125.json --dates results/analysis/temporal_minimal_pairs_event_dates_20261125.json --output <new-qwen-results.json>
```

The inference loops ran in 36.2 seconds for Qwen and 20.0 seconds for BGE on a local RTX 4060 Laptop GPU. The audit checked all synthetic gold choices against the rendered target dates and independently rebuilt all outcome totals from the saved scores.

[Qwen raw scores and pair outcomes](../results/analysis/temporal_date_binding_qwen_20261127.json) · [BGE raw scores and pair outcomes](../results/analysis/temporal_date_binding_bge_20261127.json)
