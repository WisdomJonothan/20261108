# Temporal conditions on fixed evidence pairs

The [TEMPO benchmark](https://aclanthology.org/2026.emnlp-main.291/) scores retrieval of evidence for temporally constrained queries. [MenatQA](https://aclanthology.org/2023.findings-emnlp.100/) probes temporal scope and counterfactuals in question answering. Here the outcome is whether a retriever changes the relative rank of the same two visible source passages when the requested time changes. [TimelyRAG](https://aclanthology.org/2026.findings-emnlp.398/) provides a related retrieval comparison across document versions.

## Protocol

Select eleven pairs from five official training domains. In each pair, the two original documents are positive for the same training query. Inspect their text for two distinct, clearly dated items of evidence. The manifest records the original training query ID, the passage offsets in whitespace-normalized source text, a checksum and a short explanation of each expected match. Selection precedes scoring. Each question has two variants, typically changing only “before” to “after” and otherwise preserving every word and both passages. The Scottish birth-record case changes only the requested year. The selected spans are provided directly to the models.

Measure two outcomes. A correct variant ranks the expected passage strictly first. A correct flip requires that outcome in **both** variants of a case. The original relevance annotations identify the source documents. The authored conditions and reviewed visible evidence determine the diagnostic labels. The small sample targets explicit time evidence in the official training corpus and measures behavior under controlled conditions.

## BGE pilot

| Scorer | Correct variants (22) | Correct flips (11) | Tied variants |
|---|---:|---:|---:|
| Unique query-token overlap | 8 | 0 | 6 |
| BGE-reranker-v2-m3, zero-shot | 13 | 2 | 0 |
| BGE-reranker-v2-m3, frozen task fine-tune | 12 | 1 | 0 |

The two BGE checkpoints retain the same document preference on nine and ten cases, respectively. Wrong-direction flips number zero for both checkpoints. The Scottish case asks for a source covering 1840 or 1860. The selected parish-register passage explicitly covers births in 1726–1854, and the statutory-record passage identifies the later coverage. Both checkpoints prefer the statutory-record passage for both years. In the hertz case, both checkpoints correctly switch between a 1960 adoption and a 1964–1965 adoption around the same 1963 cutoff.

This complements the [year-consistency audit](../results/TEMPORAL_CEILING.md), which found that many documents mentioning a year outside the requested interval still answer the original question. The controlled pairs hold topic and evidence availability fixed. The behavior here is a tendency to retain a topical ordering across distinct time conditions.

[Evidence manifest](../results/analysis/temporal_minimal_pairs_manifest_20261125.json) · [BGE scores and both margins for every case](../results/analysis/temporal_minimal_pairs_bge_20261125.json)

## Qwen3-Reranker-0.6B instruction control

Use the same model weights, exact passages, two query variants, official yes/no template and scoring precision. The treatment changes only the instruction. It asks the model to apply the temporal condition to the requested event, record or state, and to distinguish that date from incidental dates or later descriptions.

| Instruction | Correct variants (22) | Correct flips (11) |
|---|---:|---:|
| Default relevance instruction | 13 | 2 |
| Explicit event-time instruction | 12 | 1 |

The default and event-time instructions disagree on one of the 22 variants. The default ranks the 1887 Electoral Count Act correctly for the period after 1880. The targeted instruction instead prefers the 1877 congressional commission on that variant. The hertz pair flips correctly under both instructions and both BGE checkpoints. Simple event-time wording therefore produces a small, mixed change in score margins and leaves the dominant ranking behavior intact in this pilot. The next matched checkpoint of interest is the 8B model used for full ranking, scored with the same evidence and both presentation orders.

[Qwen scores, instructions and margins](../results/analysis/temporal_minimal_pairs_qwen_20261125.json)


## Supplied event dates with matched format control

Provide one reviewed event-time line after each original passage. Compare its correct annotation with an exchanged annotation from the other passage while keeping the same model, query variants, source text, instruction and format. The supplied dates are diagnostic annotations. They test the scorer given a normalized time cue.

| Input under the explicit event-time instruction | Correct variants (22) | Correct flips (11) |
|---|---:|---:|
| Original source passage | 12 | 1 |
| Passage plus its matched event dates | 12 | 1 |
| Passage plus the other passage's dates | 11 | 0 |

Correct date cues change which case flips. They recover the 1877 commission versus 1887 election-law contrast, while the hertz pair loses its correct early selection. The aggregate strict-flip count stays at one with correct supplied dates. The exchanged-date control changes eleven variants across six cases relative to the unannotated instruction and gives zero correct flips. The model responds to date cues, while the consistent ranking response remains rare on this fixed pilot.

[Event date annotations](../results/analysis/temporal_minimal_pairs_event_dates_20261125.json) · [Matched and exchanged date scores](../results/analysis/temporal_minimal_pairs_supplied_dates_20261125.json)


## Reproduction

Run with official v1.1 corpus files and existing local model weights. The scorer verifies every source passage against the manifest checksum and scores in CPU float32.

```bash
python scripts/eval_temporal_query_contrast.py --data-root data/track1_tempo --manifest results/analysis/temporal_minimal_pairs_manifest_20261125.json --zero-shot models/bge-reranker-v2-m3 --frozen-ce results/ce_ft/v2m3_pool8_e2/best --output results/analysis/temporal_minimal_pairs_bge_replay.json
```


```bash
python scripts/eval_temporal_query_contrast.py --data-root data/track1_tempo --manifest results/analysis/temporal_minimal_pairs_manifest_20261125.json --qwen-reranker models/qwen3-reranker-0.6b --output results/analysis/temporal_minimal_pairs_qwen_replay.json
```


```bash
python scripts/eval_temporal_query_contrast.py --data-root data/track1_tempo --manifest results/analysis/temporal_minimal_pairs_manifest_20261125.json --qwen-reranker models/qwen3-reranker-0.6b --annotated-event-dates results/analysis/temporal_minimal_pairs_event_dates_20261125.json --output results/analysis/temporal_minimal_pairs_supplied_dates_replay.json
```
