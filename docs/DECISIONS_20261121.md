# Decision record, 2026-11-21 (second phase, reviewer plan m02980)

Scope: the reviewer's next-phase list — fair reranker control, query reasoning, FollowIR p-MRR and
constraint response, error analysis, widened action space, expected-gain model, honest positioning.
All numbers are dev (519 queries, 13 domains, domain-macro nDCG@10) unless stated; every artefact is
committed and every claim goes through `scripts/check_claims.py`.

## What held up

| item | evidence |
| --- | --- |
| Frozen system re-verifies | 0.3580985233, 13 domains / 519 queries, hashes in `FROZEN_BASELINE_20261120.json` |
| Bounded tail replacement works | Direct 0.3646 (+0.0065, CI [+0.0021,+0.0117]), 51 swaps, 9,342 prompts |
| It works on other starts | dense 0.2943 -> 0.3049 (+0.0106); Qwen3-Reranker-4B top-20 start 0.2565 -> 0.2696 (+0.0131) |
| Bidirectional agreement is load-bearing | no_bidir 0.3596 vs full 0.3646 |
| Transfer to FollowIR | frozen beats dense and CE-only in all six instruction splits; Direct +0.0084/+0.0072/+0.0064 |

## What did not hold up

| item | evidence | decision |
| --- | --- | --- |
| Retained-document context | 0.3632, paired vs Direct -0.0015 at 1.77x tokens | dropped as a contribution |
| Learned gate | tau 0.0549 filtered nothing; recalibrated +0.0009 | reported as a negative result, no MLP |
| Cheap-gap budget allocation | at k=3 cheap-gap +0.0046 vs random +0.0062 | renamed and demoted to an ablation |
| Widened action space | fused -0.0009..-0.0057; dense start +0.0006..+0.0026 | not pursued |
| Allocation as the bottleneck | oracle ordering +0.0003 vs oracle selection +0.0593; model AUC 0.532 | the judgement is the bottleneck |
| Absolute condition check | 3/519 swaps, all neutral | abandoned |
| Held-out error structure | 969 train-fold queries: 5,161/8,518 fixable, band shares 44.7/34.1/21.2 | replicates dev; tail bias is real |
| Condition judge on FollowIR | deltas 0.0000/+0.0019/+0.0019; demoted removals 0/2, 2/7, 2/7 | no constraint targeting; acceptance criterion unmet |
| Learned expected-gain model | fold AUC 0.66, dev +0.0034 (CI [-0.0015,+0.0088]) | no transfer; fixed rule stays |
| Two-judge combination | agreement 0.3612 (+0.0031), veto-best 0.3616 (+0.0035) | precision for recall; no composition beats Direct |
| Cost-neutral 6/3 split | 0.3647 (+0.0066), 29/13 swaps; paired +0.0001 (CI [-0.0048,+0.0048]) | flat score, better precision |
| Evidence-first generation | 39-query sample: -0.0006, 2/2 swaps, ~4x cost, 7% unparsable | rejected on sample |
| Cascade on Direct-uncertain queries | 13 swaps, all neutral, 0.3646 at 26 prompts/query | no independent signal where Direct is unsure |
| Top-slot window (same pool, ranks 1-3) | -0.0042 (CI [-0.0214,+0.0137]), 23 helpful / 33 harmful | negative; the judge cannot overturn a confident incumbent |
| Condition-focused pair judge | fused +0.0044 (CI [+0.0012,+0.0085], 80% precision); dense start +0.0071 (CI [+0.0024,+0.0123]); reranker start +0.0034 (CI [-0.0003,+0.0074]) | kept as evidence on three starts, not deployed alone |
| Query reasoning (4B analyst) | 0.2933 -> 0.2930 | does not reproduce the published lift |
| Fair Qwen3-Reranker-4B control | single_swap +0.0035 (CI [-0.0032,+0.0105]), full_rerank -0.0408 | "not better", not "worse" |
| p-MRR on FollowIR | -0.0208..-0.0433, zero promoted documents | no instruction following yet |

## Acceptance criterion (reviewer item 7)

Not met in the strict form. The widened stage does not beat the bounded stage at equal budget on
either start, and the FollowIR evidence shows a ranking-quality gain without constraint response
(p-MRR negative, only 1/15, 5/21, 2/28 applied swaps remove a demoted document). What is established
is narrower and honest: a bounded local comparison over a small, precise candidate pool improves a
strong system by +0.0065 (CI lower bound +0.0021, 35% of that pool's ceiling) and does so on three
starting systems; the same machinery fails once the pool is widened, and the failure is located at
the judgement (AUC 0.532), not at the budget allocation.

## Positioning

SemEval 2027 RETECO system paper: the frozen system, the bounded replacement stage, the controls, the
negative results and the transfer study. Not a SIGIR/ACL method paper: that would need a comparison
signal that is reliable at the top of the ranking, which none of the four prompts tried here is.

## Next, if the project continues

1. A comparison that reads evidence spans (not whole passages) and is scored by agreement between
   two independent judges.
2. Candidate pools whose precision is controlled (CE-sourced, shallow), never top-100.
3. Distillation only after a comparison exists that beats the judge on dev at fixed budget.


## Correction, 2026-11-22: the widened-action experiment

The expanded-action harness differed from the frozen arm in two ways: prompts were
tokenised as raw text instead of through the chat template, and the reversed
presentation order stored the two option probabilities under the forward option
names. The second detail inverted the both-orders agreement rule. Re-running with
both fixed reproduces the frozen arm on the 1,686 shared comparisons (mean
|delta p| 0.0130, 98.9% same side) and reverses the result:

* fused start: +0.0082 / +0.0122 / +0.0096 / +0.0112 at 2 / 6 / 12 / 18 actions
  (4 / 12 / 24 / 36 prompts) against +0.0065 for the bounded protocol at 18 prompts;
* dense start: +0.0108 / +0.0232 / +0.0203 / +0.0170 against +0.0106 bounded;
* random subset of the whole action space: +0.0011 / -0.0017 / -0.0058 / -0.0036,
  so the pool, not the ordering within it, creates the gain;
* oracle ordering of the same judgements: +0.0258 at k=2 - the remaining headroom
  is a better prior over which pairs to ask.

Withdrawn: "widening the action space does not pay", "allocation is not the
bottleneck", the 10x rank-position cost ratio (the same-query metric ratio is
log2(11) = 3.46), the "80% precision" denominator (neutral swaps counted), the
AUC 0.532 claim (that model had three constant features), and the p-MRR sentence
that says the system ignores instruction changes (p-MRR covers dense, CE and
frozen only; Direct and the condition arm are not measured yet).

## Correction, 2026-11-22 (afternoon): capability vs evidence

The 2x2 grid fixes the candidates, the nine pairs and the rule and varies only the
judge model and the length of the passage it reads:

| judge | window | nDCG@10 | delta | helpful / harmful |
| --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct | 112 words | 0.364609 | +0.006510 | 32 / 19 |
| Qwen3-4B-Instruct | 400 words | 0.365998 | +0.007900 | 36 / 10 |
| Llama-3.1-8B-Instruct | 112 words | 0.362195 | +0.004096 | 24 / 12 |
| Llama-3.1-8B-Instruct | 400 words | 0.362623 | +0.004525 | 23 / 7 |

* more evidence, paired: +0.001390 [-0.00082,+0.00438] (4B) and +0.000429 (8B);
* stronger model, paired: -0.002414 [-0.00608,+0.00086] (112 words) and -0.003375
  (400 words), so capability does not help at either window;
* wider evidence halves the harmful loss for both judges (-2.32 to -1.28,
  -1.53 to -1.04) while cost rises 1.55x in prompt tokens and 1.8x in wall time;
* both marginal intervals include zero: this is a direction, not a result.

Decision: the binding constraint at this evidence budget is what the judge can
see, so the next experiment is evidence localisation (which span decides the
comparison and whether the 112-word query-matched window contains it), not a
bigger judge and not more prompt permutations.
