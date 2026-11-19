# Go / no-go decisions (2026-11-20)

Every threshold below was pre-registered in the project brief before the runs;
the numbers are the frozen study's own artefacts.

## Step 3 — does bounded replacement clear the bar?

Rule: keep going only if dev domain-macro nDCG@10 ≥ 0.3631 **and** the 95% CI
lower bound is > 0.

| arm | nDCG@10 | delta | 95% CI | verdict |
|---|---:|---:|---|---|
| frozen | 0.3581 | - | - | reference |
| Direct | 0.3646 | +0.0065 | [+0.0021, +0.0117] | **passes** |
| Context | 0.3632 | +0.0051 | [+0.0013, +0.0094] | passes the point estimate, adds nothing over Direct |

Paired Direct − Context: −0.0015 (CI [−0.0051, +0.0016]) at 1.77× tokens and
1.74× time. **Decision: continue with Direct; retained context is dropped as a
paper contribution.** This is the pre-registered rule, not a post-hoc choice.

## Step 4 — is the advantage real against strong controls?

| control | nDCG@10 | delta |
|---|---:|---:|
| Direct (pairwise, two orders) | 0.3646 | +0.0065 |
| same model, per-document scoring | 0.3598 | +0.0017 (ns) |
| active PRP, ≤3 pairs | 0.3626 | +0.0045 |
| Qwen3-Reranker-4B single swap | 0.3577 | −0.0004 |
| Qwen3-Reranker-4B full 13-doc rerank | 0.2671 | −0.0910 |
| Qwen3-Reranker-4B dense-100 rerank | 0.1244 | −0.2337 |

**Decision: pairwise judging beats per-document judging and every reranker
variant at comparable or lower cost; the local-replacement line continues.**
The budget curve (0.3581 / 0.3610 / 0.3655 / 0.3646 for 0–3 candidates) is
reported as quality-vs-cost, with the cap kept at 3 because cap 2 was observed
after the fact.

## Step 5 — does a learned gate pay for itself?

| rule | dev delta | calibration delta | judge calls |
|---|---:|---:|---:|
| logistic gate (τ = 0.0549) | +0.0065 | −0.0014 | 100% |
| cheap gap rule | +0.0072 | −0.0003 | 100% |
| pre-call cheap gate | +0.0011 | - | 15.6% |

The gate reproduces Direct and removes nothing; the cheap gap rule is the only
variant that improves on it (+0.0072), and it needs the judge output anyway.
A gate that runs before the judge loses 84% of the gain for 15.6% of the calls.
**Decision: no MLP, no extra training rounds, as pre-registered.** The gate is
reported as a negative result.

## Step 6 — cross-system, budget, transfer

| starting system | original | + correction |
|---|---:|---:|
| fused (frozen) | 0.3581 | 0.3646 (+0.0065) |
| dense top-100 | 0.2943 | 0.3049 (+0.0106) |
| Qwen3-Reranker-4B over dense top-20 | 0.2162 | see `exp_rr_start_dev.json` |

FollowIR could not be run: no route to Hugging Face, ModelScope or GitHub from
the server, and the official evaluation split is on none of the reachable
mirrors (probes in `docs/FOLLOWIR_TRANSFER.md`). The cross-domain table
(`SUMMARY_DEV.md`) and the three starting systems are the substitute evidence.

## Distillation

Pre-registered condition: start only if the large-model method works **and**
inference cost is the main problem, with the target of keeping ≥90% of the gain
at ≥50% less measured time.

Measured: 1.68 s and 18 prompts per query for Direct, 872 s for all 519 dev
queries on one RTX 3090. The method works, but at this scale inference is a
fixed research cost rather than the deployment bottleneck, and no trained
student can be validated without the judge labels this study is still
producing. **Decision: not started; recorded as future work with its target
written down.** This is the pre-registered rule, not an omission.
