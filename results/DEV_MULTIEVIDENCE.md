# Multi-evidence diagnostics on dev (13 domains, 519 queries)

Produced by `scripts/dev_multievidence.py` from run files that already existed;
it loads no model and re-scores nothing. `ndcg@10` is recomputed here and matches
the official aggregate to three decimals, which is what makes the other columns
comparable across systems.

## Headline

| system | nDCG@10 | relevant docs in top-10 | queries with >=2 relevant in top-10 | steps covered in top-10 | steps retrievable in top-1000 |
|---|---|---|---|---|---|
| dense mxbai top-1000 | 0.2942 | 1.03 | 31.8% | 51.3% | 96.3% |
| + zero-shot CE, z 1:1 | 0.3249 | 1.11 | 35.6% | 54.9% | 96.3% |
| + SFT CE (2 ep, frozen) | 0.3581 | 1.21 | 38.3% | 59.5% | 96.3% |

Reading: retrieval is not the bottleneck for *steps* -- 96.3% of annotated steps
have at least one gold document inside the same top-1000 pool every system
re-ranks. What separates the systems is how many of them survive into the
top 10 (51.3% -> 54.9% -> 59.5%), and how often two different relevant
documents do (31.8% -> 35.6% -> 38.3%). The frozen system is therefore
better at 'find one good passage', which is exactly the failure mode the
multi-positive loss (memo line A) is meant to attack: at 59.5% step coverage
and 38.3% multi-evidence, four out of five queries still expose a step that
the top 10 does not evidence.

## Per domain

| domain | queries | nDCG@10 | rel/top-10 | >=2 rel | step cov |
|---|---|---|---|---|---|
| bitcoin | 30 | 0.2143 | 0.87 | 26.7% | 42.9% |
| cardano | 15 | 0.2258 | 1.00 | 26.7% | 57.6% |
| economics | 25 | 0.2512 | 1.32 | 40.0% | 52.9% |
| genealogy | 35 | 0.4060 | 1.23 | 34.3% | 67.7% |
| history | 240 | 0.3918 | 1.48 | 43.8% | 66.0% |
| hsm | 45 | 0.4651 | 1.29 | 44.4% | 66.7% |
| iota | 3 | 0.8383 | 2.67 | 100.0% | 100.0% |
| law | 11 | 0.4357 | 1.18 | 45.5% | 60.9% |
| monero | 19 | 0.2069 | 0.79 | 15.8% | 46.4% |
| politics | 45 | 0.4290 | 1.18 | 42.2% | 56.3% |
| quant | 10 | 0.1538 | 0.50 | 0.0% | 46.7% |
| travel | 30 | 0.2494 | 0.97 | 33.3% | 53.8% |
| workplace | 11 | 0.3881 | 1.27 | 45.5% | 55.0% |

Source JSON: `results/DEV_MULTIEVIDENCE_{sub_sft_e2bc,dense_mxbai,sub_zs11}.json`.
Regenerate with `scripts/dev_multievidence.py --runs 'runs/<tag>_{domain}_dev.trec' --out ...`.

## What the annotation expects

Computed directly from the dev qrels and step qrels (519 queries, 13 domains):

* **2.34 steps per query** (median 2) and **3.44 judged-relevant documents per
  query** (median 3);
* **78.8%** of queries are annotated with two or more steps and **87.9%** have
  two or more relevant documents;
* the frozen system puts **1.21** relevant documents in its top 10 and reaches
  two or more for **38.3%** of queries.

So the gap is not "the corpus does not contain the evidence" (96.3% of steps
are retrievable) and not "the query has a single right answer" (79% need more
than one step). It is that the ranker learns to surface the single easiest
passage and stop -- the exact failure mode `L_step` (memo line A) is designed to
put pressure on, with `step_hit_rate` / `multi@10` as the diagnostics that must
move if the intervention works.
