# Fixed-budget semantic window selection

The previous [supervised selection experiments](SWAP_RISK_FOLLOWUP_20261124.md) reused ranking and judge scores. This experiment supplies different passages to the judge while retaining the candidate IDs, six head-first comparisons, Direct prompt, two presentation orders, and maximum one replacement.

The reference passage is the 112-word window with the highest lexical overlap. The alternative uses the frozen BGE-reranker-v2-m3 to select one window from the eight highest-overlap windows of each document, using the full normalized question truncated at 192 question tokens. The selector uses sequential inference before the Qwen judge. The alternative supplies up to the same number of words per document, although token counts and total computation differ. Identical excerpts reuse their original score in both policies.

| Policy on training fold f0 | Domain-macro nDCG@10 | Helpful / harmful swaps |
|---|---:|---:|
| Original lexical windows | 0.259758 | 20 / 7 |
| BGE semantic window selection | 0.256400 | 15 / 10 |
| Only try the semantic judgment when the lexical policy abstains | 0.257722 | 23 / 12 |
| Require the same action to be accepted under both window policies | 0.258481 | 12 / 5 |

This is an out-of-fold pilot on 328 training queries, not official dev. The primary difference is −0.00336 (paired within-domain bootstrap 95% interval [−0.00684, +0.00016]). The evidence selector scores 8,456 question–window pairs and changes at least one excerpt on 1,372 of the 1,968 judged actions. The repeated Qwen scoring then evaluates 2,744 prompts in 247 seconds. The two combined-policy rows replay both caches and use extra comparison budget. They are explanatory controls.

Among the changed-excerpt actions, nine useful actions become acceptable but nineteen useful actions stop being acceptable. Eleven harmful actions become acceptable and six stop being acceptable. The effect is not just a decision about which eligible action to take. Window choice itself changes which relevant passages the judge will accept. A frozen document-relevance score does not provide a dependable way to select concise evidence that preserves the judge's best decisions.

Keep the competition configuration at 0.373503 on official dev. The f0 result does not justify expanding this selector to the rest of training or to dev. The next scope is recovering judged positives in the 78 pool-reachable zero-score dev queries, with a stage-specific control for candidate prioritization versus actual text-based judging.

The [next candidate-budget follow-up](CE_DISAGREEMENT_FOLLOWUP_20261124.md) evaluates a CE disagreement policy and a fixed-pool retrieval control.

[GPU comparison](../results/analysis/semantic_judge_windows_f0_20261124.json) · [Combined-policy replay](../results/analysis/semantic_judge_window_replay_f0_20261124.json)

```bash
python scripts/eval_semantic_judge_windows.py --selector models/bge-reranker-v2-m3 --judge models/Qwen3-4B-Instruct-2507 --fold f0 --output results/analysis/semantic_judge_windows_f0_20261124.json
python scripts/replay_evidence_window_support.py --input results/analysis/semantic_judge_windows_f0_20261124.json --reference results/analysis/position_oracle_oof_20261124.json --output results/analysis/semantic_judge_window_replay_f0_20261124.json
```
