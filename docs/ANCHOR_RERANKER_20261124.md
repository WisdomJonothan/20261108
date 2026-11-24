# Shared-reference relevance scoring

The preceding [fixed-budget experiments](CE_DISAGREEMENT_FOLLOWUP_20261124.md) improved the number of promising actions more often than the final list. This study measures the underlying Qwen pairwise signal over a broader fixed document pool. It is a pilot for a stronger ranking stage.

For each query, compare each of the other ninety-nine dense candidates against rank five of the original fused list. Use 112-word lexical excerpts, the existing Direct prompt, and both presentation orders. The document's score is the mean log odds of choosing the candidate over the reference. Query-level standardized scores are evaluated alone and in a fixed 30% blend with the original fused score. No labels choose the anchor or weight.

Shared reference scoring has direct prior work in [RefRank](https://arxiv.org/abs/2506.11452); pairwise LLM prompting also has established retrieval baselines such as [PRP](https://aclanthology.org/2024.findings-naacl.97/). Here the research question is whether the same judge provides a better ranking signal when it sees the entire existing pool instead of selecting a single replacement, and how much of the gain survives a tighter pool.

## First held-out sample

Two queries per domain were sampled from the 969 existing held-out training-fold queries. These are 26 queries and no official dev queries.

| Ranking on the same 26 queries | nDCG@10 | Qwen prompts per query |
|---|---:|---:|
| Original fused ranking | 0.25880 | 0 |
| Six-pair external replacement | 0.25941 | 12 |
| Shared-reference scores alone | 0.28489 | 198 |
| Original scores with 30% shared-reference score | 0.28512 | 198 |
| Same blend, protected top ten and ten dense outsiders | 0.28426 | 38 (cache replay) |

Against the six-pair policy the full blend gains +0.02571. The within-domain query bootstrap interval is [+0.00422, +0.04759] and the domain-resampling interval is [−0.01518, +0.06925]. Eight of thirteen domains improve, four decline, and one ties. The full blend recovers two original zeroes and creates none. Only two queries per domain constrain these uncertainty estimates. The cheaper pool is an exploratory cache replay and would require thirty-eight prompts if executed directly.

## Disjoint replication and smaller pool

The follow-up excluded all twenty-six query IDs before drawing three new held-out queries per domain. Anchor rank, fusion weight, and presentation orders remain fixed.

| Evaluation | Original fused | Six-pair | Anchor alone | Full-pool fusion | 20-document fusion replay |
|---|---:|---:|---:|---:|---:|
| First 26 | 0.25880 | 0.25941 | 0.28489 | 0.28512 | 0.28426 |
| New 39 | 0.28361 | 0.27666 | 0.24061 | 0.28830 | 0.29810 |
| Combined 65 | 0.27369 | 0.26976 | 0.25832 | 0.28703 | 0.29256 |

The standalone reference score falls on the second sample. The 20-document 70:30 blend improves both samples over their matched six-pair results (+0.02485 and +0.02144). The pooled paired within-domain interval is [+0.01038, +0.03563]. The 65-query full-pool blend has a domain-resampling interval of [−0.01385, +0.04976] relative to six-pair. Small strata limit inference. Annotated evidence step coverage on the two samples combined is 57.85% for six-pair and 57.79% for the full reference blend.

The direct official-dev test is running with a fixed protocol: preserve the reference top ten, add ten highest dense outsiders, use the original rank-five anchor, and fuse scores at 70:30. All 519 queries receive 38 prompts each. The project will keep the 0.37350 six-pair configuration as the reference when reporting nDCG and cost.

[Disjoint raw results](../results/analysis/anchor_reranker_oof_replication39_20261124.json) · [Paired sample summary and 20-document replay](../results/analysis/anchor_reference_cap20_two_samples_20261124.json)

[Raw first-sample score cache](../results/analysis/anchor_reranker_oof_sample26_20261124.json) · [Same-query policy comparison](../results/analysis/anchor_reference_domain_sample26_20261124.json)

## Independent 0.6B reranker control

Cached Qwen3-Reranker-0.6B scores receive 25% weight in a standardized blend on the same 519 official-dev top-100 pools. This changes 0.35810 to 0.35607. A separate check vetoes the frozen six-pair decision when this 0.6B reranker prefers the incumbent. It removes fourteen helpful swaps and one harmful swap, changing 0.37350 to 0.36633. This independent model does not make a useful veto. [Dev replay](../results/analysis/independent_reranker_signal_20261124.json)

The official-dev configuration remains at 0.37350. The reference-based results above come from the two training-fold samples.

```bash
python scripts/eval_anchor_reranker.py --model models/Qwen3-4B-Instruct-2507 --queries-per-domain 2 --output results/analysis/anchor_reranker_oof_sample26_20261124.json
python scripts/eval_anchor_reranker.py --model models/Qwen3-4B-Instruct-2507 --queries-per-domain 3 --exclude-report results/analysis/anchor_reranker_oof_sample26_20261124.json --output results/analysis/anchor_reranker_oof_replication39_20261124.json
python scripts/eval_anchor_reranker.py --model models/Qwen3-4B-Instruct-2507 --split dev --pool-depth 20 --output results/analysis/anchor_cap20_official_dev_20261124.json
```
