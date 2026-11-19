# RETECO research update · 2026-11-19

The reference system retrieves with mxbai-embed-large-v1 and reranks with task-fine-tuned BGE-reranker-v2-m3. Its official-dev domain-macro nDCG@10 is **0.3581**, versus **0.3249** for the matched zero-shot pipeline. Our research asks why a stronger standalone reranker can produce a weaker fused ranking, and whether a bounded replacement policy can recover relevant documents lost from the final top ten.

## Fusion results on official dev

All rows use the same dense top-100 candidate pools on 519 queries from 13 domains. Within a seed, both query formats use matched training settings. Each calibrated weight is selected on the internal training holdout.

| Seed | Title CE alone | Full CE alone | Title equal fusion | Full equal fusion | Title calibrated | Full calibrated |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 0.29237 | 0.33096 | 0.35055 | 0.34502 | 0.34270 (0.7) | 0.36422 (0.7) |
| 18 | 0.29220 | 0.31842 | 0.34781 | 0.34555 | 0.35228 (0.6) | 0.34081 (0.9) |
| 19 | 0.29299 | 0.33285 | 0.34917 | 0.34794 | 0.34917 (0.5) | 0.35743 (0.6) |

Parentheses give the CE weight. The full-query model improves standalone ranking in all three seeds, while the equal-weight fusion orders the two query formats the other way around. The seed-17 score of 0.3642 is the highest observed calibrated run. The mean across the three independently calibrated full-query runs is 0.3542. Reusing seed 17's 0.7 weight for seed 18 gives 0.3565. In repeated internal two-fold evaluation, 50% shrinkage toward equal weighting reduces the loss associated with direct weight selection and scores near equal weighting.

For a pair containing a qrels-positive and a benchmark-zero document, full-query CE reverses dense-correct pairs about 4.0–5.0 percentage points less often than title CE. After equal fusion, the full-query system corrects about 4.0–4.4 percentage points fewer dense errors on pairs spanning the dense top-ten boundary. These are paired diagnostics on a fixed pool. Unjudged documents count as zero under the evaluation convention. Individual domain effects differ. Doubling the weights of active LambdaRank pairs targeted at this boundary changed fused nDCG@10 by +0.00026, with a domain-bootstrap 95% interval of [−0.00086, +0.00152].

Artifacts include [pair diagnostics](../results/pair_correction_analysis_s19.json), [depth and RRF controls](../results/fusion100_posthoc_depth_rrf_diagnostic.json), [calibration stability](../results/stable_calibration_crossfit.json), and the [boundary training audit](../results/boundary_intervention_audit.json).

## Limited replacement

In a dev oracle analysis starting from the 0.3581 reference system, incoming candidates come from the union of dense and full CE top-20, excluding the reference top ten. With perfect qrels-based choices, one swap at any top-ten position reaches 0.5104. Restricting the swap to ranks 8–10 lowers that ceiling to 0.4098. The lexical proxy reached 0.35736 against the original 0.35810. An E5 semantic proxy also scored close to its dense-only internal baseline. [Oracle results](../results/restricted_replacement_oracle.json) · [Lexical pilot](../results/replacement_pilot.json) · [E5 pilot](../results/semantic_replacement_pilot.json)

The Qwen3-4B-Instruct-2507 experiment starts from the dense-only top ten on the **214-query, 13-domain training holdout**. Each query has three fixed document pairs: dense ranks 11/10, 12/9, and 13/8. Both prompts score each pair twice with reversed presentation. A policy averages identity-aligned log probabilities and applies at most one swap with a positive margin over both keeping the incumbent and abstaining. The context prompt reads extra excerpts from retained results. Labels are used to score outcomes.

| Policy | Domain-macro nDCG@10 | Helpful / harmful swaps | Domain-macro step coverage |
|---|---:|---:|---:|
| Dense starting ranking | 0.23182 | — | 43.38% |
| Direct pair comparison | 0.24494 | 21 / 6 | 46.56% |
| Pair comparison with retained evidence | 0.24331 | 18 / 5 | 46.88% |

The direct policy improves this dense baseline by +0.01312 (exploratory domain-bootstrap 95% interval [0.00101, 0.02901]). Adding retained evidence changes the direct score by −0.00163 (interval [−0.00676, 0.00205]) and improves step coverage by 0.32 percentage points. This measures a fixed replacement policy on an internal development split. The 0.3581 reference is an official-dev result from a different starting system. [Per-pair scores](../results/pairwise_replacement_qwen4b_holdout214.json) · [Per-query results and uncertainty](../results/pairwise_replacement_qwen4b_holdout214_summary.json)

## Next experiment

Apply the paired controls to the reference system's top ten on the same evaluation queries. Fix the source ranks, candidate documents, eligible slots, prompt budget and one-swap limit before scoring. Evaluate the unchanged ranking, direct comparison, and retained-evidence comparison. Keep the paired inference calls identical across the comparison arms and record input-token cost. Report nDCG@10, harmful and helpful swaps, and step coverage separately. This gives an estimate of the gain from applying the judge and a separate estimate of the gain associated with evidence context.

The paper's established result is the replicated fusion reversal and its pair-level analysis. A method claim about retained evidence depends on the paired control above. A stronger standalone ranking is a separate system experiment. Compare each starting system with and without replacement to measure the incremental contribution of the policy.

## Reproduction

Place the official corpus and qrels in `data/track1_tempo`. For the local Qwen checkpoint, use the public [Qwen3-4B-Instruct-2507 weights](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

```bash
python scripts/audit_restricted_replacement.py
python scripts/eval_replacement_pilot.py
python scripts/benchmark_pairwise_replacement.py --model models/Qwen3-4B-Instruct-2507 --queries-per-domain 214 --pairs-per-query 3 --output results/pairwise_replacement_qwen4b_holdout214.json
python scripts/summarize_qwen_replacements.py
```

Run the commands from the repository root with the dependencies described in [REPRODUCE.md](REPRODUCE.md). The Qwen summary can also be regenerated directly from the shipped per-pair scores, dense training runs and official annotations.

The included `*_dev_head10.trec` and `*_train_head20.trec` files are rank prefixes for these experiments. Full training runs and candidate groups are needed for training and the boundary intervention audit.
