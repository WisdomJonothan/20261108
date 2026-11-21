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
