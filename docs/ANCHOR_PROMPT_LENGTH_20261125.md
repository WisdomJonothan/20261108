# Oversized tokens in the anchor reranker inputs

The depth-100 official dev run encountered a CUDA memory failure with 316 queries saved. The failed history query has 408 query tokens. Three of its selected 112-word document windows each contain a continuous sequence of 1,065,178 identical dash punctuation characters. Two orders per pair produce six 17,853-token prompts. The query completed at batch size one on the same GPU and checkpoint.

Scan the selected windows across all 519 official dev queries. Count a whitespace-delimited word as oversized if it has at least 2,048 characters. Audit the prompt lengths in both orders before and after shortening only such words to their first 256 characters. Both arms use the same original text selection and tokenizer.

| Quantity | Measured value |
|---|---:|
| Queries with an oversized selected word | 27 |
| Affected selected windows | 42 |
| Affected judge prompts | 84 |
| Original tokens across affected prompts | 312,045 |
| Tokens across affected prompts after word clipping | 51,400 |
| Longest affected prompt, original | 62,010 |
| Longest affected prompt with clipping | 989 |

The largest selected string contains 3,307,683 identical dash punctuation characters in an HSM candidate. The tokenizer converts this one whitespace-delimited word into most of a 62,010-token prompt. This accounts for a large peak input even when the query and evidence window have ordinary word counts. The total reduction across affected prompts is 260,645 tokens. This audit measures tokenizer length.

## Ranking checksum with clipped inputs

Rescore the 84 affected prompts with the same 8B judge and retain the cached scores of all other prompts. This replay processes 51,400 tokens in 14.03 seconds with 16,118 MiB peak PyTorch allocation on the RTX 3090. It changes top-ten membership for one history query (69860_322). All 519 per-query nDCG@10 values agree with the original, yielding the same 0.37959 domain-macro nDCG@10. The main full-run score comes from the original prompt text. [Rescored pairs and rankings](../results/analysis/anchor_trim_replay_dev_20261125.json)

[Per-query prompt lengths and string distributions](../results/analysis/anchor_prompt_length_audit_20261125.json)

```bash
python scripts/audit_anchor_prompt_outliers.py --model models/Qwen3-8B --output results/analysis/anchor_prompt_length_audit_replay.json

python scripts/eval_anchor_trim_replay.py --audit results/analysis/anchor_prompt_length_audit_20261125.json --direct-report results/analysis/anchor_reranker_qwen8b_dev100_20261125.json --model models/Qwen3-8B --output results/analysis/anchor_trim_replay_check.json
```
