# Adversarial audit — `results/replacement_v2` (RETECO Track-1a dev, 519 queries / 13 domains)

**Reviewer role:** hostile reviewer. Everything below is either reproduced from raw inputs or quoted with a file:line. Nothing was modified outside `results/audit/`.

**Environment:** Windows, Git-Bash, `D:/anaconda/python.exe`, `PYTHONUTF8=1`, repo `D:\RETECO\github_repo`.

**Bottom line.** The core empirical claim is *real and reproducible*: I rebuilt the frozen ranking metric, the candidate pool and the one-swap ceiling from the raw TREC runs + qrels with no committed derived artefact, and got 0.3580985233 / pool mismatch 0 / 0.3769401158. What is **not** defensible as written: (i) the shipped `random`-budget curve is not reproducible and the paper's numbers for it match nothing in the tree; (ii) the widened-action results and the oracle ladder built on them have **no producer script in the repo** and cannot be regenerated; (iii) the report's own Conclusion still calls that widened space a *failure* with numbers that no longer exist; (iv) the headline is one of ≥120 configurations evaluated on the same dev set, and the "+0.005 pre-registered bar" it is said to clear was committed **2h18m after the result it judges**. Individually each is fixable in a day; together they are the difference between "a careful small result" and "a claim that collapses under an artifact-evaluation request".

---

## Verdict table (the eight requested checks)

| # | Check | Verdict | One-line reason |
|---|---|---|---|
| 1 | `replay_replacement.py --quick`, `check_claims.py` | **PASS** | Both exit 0; replay prints `RESULT: replay matches every committed number`; 60+ claim checks `ok`. Caveat: it starts from stored probabilities, not from the model (see F7). |
| 2 | Independence (no qrels in generation, judge blind, `decide()` on predictions only) | **PASS** | `build_replacement_candidates.py` never opens qrels; prompt builder reads only query + two passages; `decide()` reads only `pred.fwd/pred.rev`. Latent qid-keying residue in replay (F10). |
| 3 | Metric integrity (13 equal-weight domains, missing = 0, reproduce a domain) | **PASS** | Independent rebuild: all 13 per-domain values + macro 0.3580985233 exactly (0 missing queries). |
| 4 | Multiple comparisons / threshold gaming | **FAIL** | ≥120 dev-evaluated configurations; CI-implied p for the headline is 0.0083, so it survives Bonferroni only if the family is ≤6. Nothing reproducible survives m=120. |
| 5 | Statistical validity (within-domain paired bootstrap, seed reproducibility) | **PASS (mechanics)** | Resamples within domain; identical output on re-run with seed 20261120; stable to 1.4e-5 under seed 12345. But `p_le_0` is resolution-limited/lumpy and must not be used as a calibrated p-value (F6). |
| 6 | Ceiling claims (0.3769 ladder) | **PASS for the capped ceiling / FAIL for the widened ladder** | 0.3769401158, +0.0188415926, 68/519 reproduced from raw runs. The widened-space ladder rows are unreproducible and one row is mislabelled (F2, F8). |
| 7 | Train/dev leakage | **PASS** | dev ∩ fold-train = 0/519; dev ∩ `ce_folds.fold_of` = 0; dev ∩ `fusion100_split.{exclude,eval}` = 0. Residual: CE val ids not regenerable (disclosed). |
| 8 | Reporting hygiene (stale numbers) | **FAIL** | Stale Conclusion §10 contradicting the corrected §6c; random-budget numbers in README/paper match no artefact and no run; "every CI below 0" sign error. |

---

## A. Blocking findings

### F1 (Critical, reporting integrity) — the report's Conclusion still calls the widened space a failure, with numbers that exist nowhere

`results/replacement_v2/REPORT.md:456-466`:

> "Retained context, a learned gate, cheap-gap allocation, a widened top-100 action space, absolute condition checks, condition-focused pairs, query reasoning and the off-the-shelf reranker control **do not add to that** … The **widened-space failure** is located at the judgement (AUC 0.532; oracle selection +0.0593 versus oracle allocation +0.0003), which is where any further work has to start."

The same file, in §6c/§6d2 (`REPORT.md:115-155`), reports the opposite for the corrected widened run:

| k | 2 | 6 | 12 | 18 |
|---|---|---|---|---|
| delta (prior order) | +0.0082 | +0.0122 | +0.0096 | +0.0112 |
| 95% CI lower | +0.0027 | +0.0055 | — | — |

Verified against `results/replacement_v2/expanded3_fixed_policies.json`: k=2 `+0.008241770789422646` CI `[+0.002735837857663795,+0.015270251533200713]`; k=6 `+0.012150580105915881` CI `[+0.005453768767327447,+0.020295010977485286]`; k=12 `+0.009621633057183773`; k=18 `+0.01123933911136931`. Every one is larger than the headline +0.00651.

Evidence that the §10 numbers are stale, not merely different:
```
$ grep -rn "0\.0593\|0\.532" results/replacement_v2/*.json
results/replacement_v2/candidates_dense_start_dev.json:3484: "incoming_dense_gap": -0.05937400000000004,
...   # only incidental -0.0593 gaps; no artefact stores 0.0593 / 0.0003 / 0.532
```
The §10 text is dated `(2026-11-21)` and was not updated by commit `b3c34e3` (2026-11-22 03:43:20, "…and replace the invalidated results"), which rewrote §6c/§6d2 and README but not this paragraph. A reviewer who reads the conclusion and then the section it summarises will conclude the authors do not know which of their own results is current. **This alone is enough to reject on presentation.**

Fix: rewrite §10 to match §6c/§6d2, or delete the "do not add" / "failure" sentence and the 0.0593/0.0003/0.532 decomposition, which cannot be traced to any committed artefact.

### F2 (Critical, reproducibility) — the widened-action results (and the oracle ladder over them) have no producer in the repository

Every committed file for the top-100 / ten-slot experiment is a *result*; the script that produced it was never committed, before or after the 2026-11-22 correction:

```
$ git log --diff-filter=A --format='%h %ad %s' --date=iso -- scripts/benchmark_pairwise_replacement_v2.py
c2e4f84 2026-11-20 14:36:56 +0800 Reproducibility: ship run scripts, ...
$ git log -1 --format='%h %ad %s' --date=iso 671cbcc        # headline artefact commit
671cbcc 2026-11-20 02:53:32 +0800 Record dev replacement results: Direct +0.0065 ...
```
so even the *narrow* harness postdates the numbers it is supposed to have produced (see F7). For the widened harness:
```
$ grep -rln "oracle_order" --include=*.py .        →  ./scripts/check_claims.py   (reader only)
$ grep -rln "18-action\|eighteen" --include=*.py . →  (nothing)
$ git log --all --diff-filter=A --name-only -- "*action*"  →  (nothing)
```
`b3c34e3` added `expanded3_fixed_dev.json`, `expanded3_fixed_policies.json`, `expanded3_fixed_random_dev.json`, `expanded3_fixed_densestart_dev.json` (429,833 lines each) and `cmp_fixed_vs_direct.txt` — **and no script**. `scripts/check_claims.py`, `report_v2.py`, `ea3_summary.py`, `gate_ea3.py` only *read* these files.

`expanded3_fixed_dev.json` does store per-query deltas, so the *statistics* can be audited (I replayed its bootstrap and reproduced point/CI/`p_le_0` exactly for all four budgets — it is internally consistent). But the **decisions themselves** (which of 18 actions, per query, at each k) cannot be re-derived from anything in the tree: there is no action cache, no stored two-order probabilities for the widened run, and no code. A reviewer must take 0.36634/0.37025/0.36772/0.36934 and the +0.0258/+0.0212 oracle-ordering numbers **on trust**, and the paper's strongest post-fix paragraph rests on them.

Fix: commit the widened-run harness (and ideally the 18-action probability cache, as was done for Direct in `exp_direct_probs_dev.json`), or demote the widened results to "reported, not reproducible" and drop them from the claims. Note also that the doc's own labelling — `expanded3_fixed_policies.json` "note": "replay over the corrected 18-action cache" — presupposes a cache file that is not in the repo.

### F3 (Critical, reproducibility + stale numbers) — the `random` budget curve is not reproducible, and the numbers in README/paper match neither the artefact nor any run

`scripts/budget_from_cache.py:114`:
```python
rng = random.Random(SEED + hash(q["qid"]) % 100000)
```
Python's `str.__hash__` is salted per process (`PYTHONHASHSEED`), so the per-query RNG stream changes between processes. Three independent draws, same command, same SEED:

```
$ D:/anaconda/python.exe scripts/budget_from_cache.py --k 1 3 9 --out results/audit/runA.json
$ D:/anaconda/python.exe scripts/budget_from_cache.py --k 1 3 9 --out results/audit/runB.json
```

| random ordering | k=1 | k=3 |
|---|---|---|
| committed `budget_cache_dev.json` | **+0.001659** | **+0.006189** |
| `results/audit/runA.json` | **+0.000876** | **+0.000856** |
| `results/audit/runB.json` | **+0.002451** | **+0.004365** |
| README.md:155-157 / paper/main.tex:176 | **+0.0037** | **+0.0060** |

All other orderings (fixed/cheapgap/uncertain/oracle) are bit-identical across A, B and the artefact, so this is isolated to the `hash()` seed. The k=3 spread (0.00086 → 0.00619) is the same size as the headline effect.

Consequences:
1. The paper's sentence "the gap-based order does not beat a random pair at these budgets" (`paper/main.tex:176-180`) is **draw-dependent**: at k=3 the artefact gives random 0.00619 > cheapgap 0.00464, run A gives 0.00086 < 0.00464, run B gives 0.00437 < 0.00464. Two of three draws contradict the paper's sentence.
2. The paper's `+0.0037 / +0.0060` match **no** committed artefact and **no** fresh run. This is a stale number, exactly the class of defect the task calls a FAIL.
3. The same defect makes the `random` row of `expanded3_fixed_policies.json` suspect unless that harness used an explicit seed per pair (the producer is absent, so this cannot be checked — F2).

Fix: replace `hash(qid)` with a stable hash (`zlib.crc32(qid.encode())` or the saved string), regenerate `budget_cache_dev.json`, then correct README/paper from the regenerated artefact. Note the qualitative conclusion may need to change: a fair random draw does not reliably lose to the gap ordering at k=3.

---

## B. Major findings

### F4 (Major, statistics) — ≥120 configurations on one 519-query dev set; the headline survives Bonferroni only as a family of ≤6

Counted from the committed tree (each cell is a distinct decision evaluated on the same 519 dev queries):

| source | configurations |
|---|---|
| `exp_*.json` (26 files) | 28 arms (direct, contextual, perdoc, active, cap1, cap2, dense_start, direct_top3, evidence_sample, llama31, llama31_wide, perdoc, rr20, train_f0/f1/f2/fall, two_win, violation, violation_cascade, violation_pair, violation_pair_dense, violation_pair_rr, wide_win400, cond_win, condition_pairs, combined_pair, direct_probs[=direct duplicate]) |
| `veto_rules_dev.json` | 7 thresholds |
| `gate_eval.json` | 5–6 rules |
| `ablations_dev.json` | 4 |
| `ensemble_rules_dev{,_either}.json` | 2 |
| `hybrid_rules_dev.json` | 5 |
| `split_k6.json` / `split_k9` | 6 |
| `budget_cache_dev.json` | 5 k × 5 orderings = 25 |
| `expanded3_fixed_policies.json` (+ dense-start/random/prior variants) | 20 (+≈60) |
| FollowIR collections | 3 |

≥120 (≈180 counting the expanded3 variants). Reported bootstrap p is `p_le_0 = 0.0010`; the CI-implied normal p and the family size each claim would need:

| claim | delta | CI | implied SE | z | two-sided p | survives Bonferroni only if m ≤ |
|---|---|---|---|---|---|---|
| Direct (headline) | +0.00651 | [+0.00206,+0.01172] | 0.00246 | 2.64 | **0.0083** | **6** |
| Context | +0.00505 | [+0.00126,+0.00943] | 0.00208 | 2.42 | 0.0153 | 3 |
| cap2 | +0.00744 | [+0.0026,+0.0131] | 0.00268 | 2.78 | 0.0055 | 9 |
| wide400 | +0.00790 | [+0.0030,+0.0135] | 0.00268 | 2.95 | 0.0032 | 16 |
| `ce_or_dense_gap` | +0.0072 | [+0.0030,+0.0123] | 0.00237 | 3.03 | 0.0024 | 21 |
| hybrid (cond⊕direct) | +0.00802 | [+0.00296,+0.01383] | 0.00277 | 2.89 | 0.0038 | 13 |
| expanded3 prior k=6 | +0.01215 | [+0.00545,+0.02030] | 0.00379 | 3.21 | 0.0013 | 38 |
| condition_pairs | +0.00443 | [+0.00118,+0.00851] | 0.00187 | 2.37 | 0.0180 | 3 |

- At **m = 120** (α/m = 4.2e-4) **no reproducible claim survives**; the only cells with `p_le_0 = 0.0` are the unreproducible expanded3 ones (F2).
- At the most charitable confirmatory family — the six full-dev same-start arms ≥ +0.005 identified in F5 — α/6 = 8.3e-3, and the headline (p = 0.00825) sits *exactly on the line*: m = 6 passes by 1e-4, m = 7 fails. The only claim with room to spare is `ce_or_dense_gap` (p = 0.0024, survives m ≤ 21), which is a dev-selected gate variant, not the paper's headline. This knife-edge is the crux: the paper's statistical claim depends entirely on how the family is declared.
- The headline therefore survives only under the "single pre-registered primary endpoint" defence (m=1) or a family of at most six. That defence is undermined by F5 (six same-start arms exceed the headline) and by the bar's commit date:

```
$ git log -1 --format='%h %ad %s' --date=iso f180fdf   # froze pool + ceiling + baseline
f180fdf 2026-11-20 01:22:20 +0800 Freeze replacement protocol: re-verified 0.3581 baseline, cap-3 candidate pool, 0.3769 ceiling
$ git log -1 --format='%h %ad %s' --date=iso 671cbcc   # headline artefact
671cbcc 2026-11-20 02:53:32 +0800 Record dev replacement results: Direct +0.0065 ...
$ git log --diff-filter=A --format='%h %ad' --date=iso -- configs/replacement_study_20261120.json
cacb8e1 2026-11-20 05:11:01       # the file holding the "+0.005 AND CI>0" bar
```

The **pool, slots and ceiling were pre-registered** (01:22 < 02:53) — that part of `paper/main.tex:28` survives. The **acceptance bar was committed 2h18m after the result it judges** (`configs/replacement_study_20261120.json:77`), so `paper/main.tex:173` "Direct clears the pre-registered bar of +0.005" is unsupported by git order. It happens to pass the bar anyway; the problem is the word *pre-registered*, not the threshold.

Fix (choose one, in the paper and README): (a) drop "pre-registered" for the bar and describe it as the acceptance criterion used in the written protocol; or (b) point to `f180fdf` and state the bar there; or (c) add a sentence: "the pool and ceiling were frozen before the result; the +0.005 acceptance threshold was recorded in the protocol file after the first run, and Direct exceeds it by 0.0015". Option (c) is honest and costs one sentence. Also add a limitations paragraph naming the ≥120 configurations and quoting the Bonferroni arithmetic above.

### F5 (Major, selection) — the headline arm is not the best observed, and at least eight same-start configurations beat it

Full-dev, same frozen start (0.3580985233), same 519 queries and same pool; I verified each number against the named file:

| configuration | delta | paired vs Direct | source |
|---|---|---|---|
| expanded3 prior k=6 | **+0.01215** | — (different action space) | `expanded3_fixed_policies.json` (unreproducible, F2) |
| hybrid `either` (cond⊕direct) | **+0.00802** | **+0.00019, CI [−0.00011,+0.00435]** | `hybrid_rules_dev.json` (`either`, `paired.either_minus_direct`) |
| wide400 (400-word window) | **+0.00790** | +0.0014, CI [−0.0008,+0.0044] | `exp_wide_win400_dev.json`; paired in `REPORT.md:447-448` |
| cap2 | **+0.00744** | +0.00093 | `exp_budget_cap2.json` |
| `ce_or_dense_gap` gate | **+0.0072** | +0.0007 | `gate_eval.json` |
| split 6 relevance + 3 condition | **+0.00657** | +0.0001, CI [−0.0048,+0.0048] | `split_k6.json` |
| **Direct (headline)** | **+0.00651** | — | `exp_direct_context_dev.json` |
| fixed-order k=6 (subset of the same arm) | +0.00661 | — | `budget_cache_dev.json` |
| random-order k=6 (subset of the same arm) | +0.00674 | — | `budget_cache_dev.json` (irreproducible, F3) |
| cond_win (112-word condition window) | +0.00639 | +0.00047, CI [−0.0006,+0.0016] | `exp_cond_win_dev.json` |
| ~~`violation_pair_dense`~~ +0.00713 | — | dense start 0.2943, not the frozen start | `exp_violation_pair_dense_dev.json` |
| ~~`train_f2`~~ +0.00684 | — | 318-query fold subset, base 0.3098 | `exp_train_f2.json` |

So six full-dev configurations on the frozen start exceed the headline, and two cheaper allocations of the same arm do too. The saving grace is real and the paper should lead with it: **every paired interval against Direct includes zero** (+0.00019 for the best, +0.0014 for 400 words), so no variant is *established* as better — but the paper currently does not say that anywhere. Without that sentence a reviewer reads the appendix, finds six larger numbers, and asks why the paper is not about the gate or the 400-word window. Recommended: "Several post-freeze variants score higher on dev (up to +0.0122 in a widened action space); none is reported as a contribution because it was selected after the protocol was frozen and its paired interval against Direct includes zero (400-word window +0.0014, CI [−0.0008,+0.0044]; condition⊕relevance +0.0002, CI [−0.0001,+0.0044])." That converts F5 into evidence of discipline. Note this also removes the "pre-registered protocol ⇒ single primary endpoint" defence used in F4: if six arms exceed the headline, calling the pooled family m=1 is a choice, not a fact.

### F6 (Major, statistical reporting) — `p_le_0` is not a p-value, and the CI is driven by ~51 non-zero queries

- Mechanics PASS: `paired_bootstrap` in `scripts/benchmark_pairwise_replacement_v2.py` builds `deltas[domain] = np.array([...])` and resamples `v[rng.integers(len(v), size=len(v))]` per domain, then `np.mean([...])` over 13 domains — correct for a domain-macro estimand. Re-running with seed 20261120 reproduces point 0.0065102557 / CI [0.0020642841, 0.0117240276] / p 0.0010 exactly; seed 12345 gives [0.0020508201, 0.0117593336] — stable.
- But 195 no-swap + 273 neutral swap counts mean **468 of 519 per-query deltas are exactly 0**; only 51 queries (32 positive, 19 negative) carry any mass. The bootstrap distribution is discrete and lumpy, `p_le_0` is floored by resolution (10 hits in 10,000 draws), and the normal-approx p implied by the CI is 0.0083, eight times the reported 0.001. Reporting `p = 0.001` in the text as if it were a calibrated p-value is misleading; report the CI and, if a p is needed, a randomisation/permutation p over the 51 non-zero queries or a sign-flip test, with the domain structure preserved.
- Also note the estimand mismatch that the paper handles correctly but a reader may trip on: the query-level mean difference Direct−Context is −0.00128 while the domain-macro difference is +0.00151, because domains are weighted equally. The paper reports the domain-macro version (+0.0015, CI [−0.0051,+0.0016], p=0.81); the query-level sign is opposite. Say so explicitly.

### F7 (Major, provenance) — the harness that produced the headline was committed 11.7 h after the result and has been edited three times since; nothing ties it to the artefacts

```
$ git log --diff-filter=A --format='%h %ad %s' --date=iso -- scripts/benchmark_pairwise_replacement_v2.py
c2e4f84 2026-11-20 14:36:56 +0800 Reproducibility: ship run scripts, raw two-order probabilities, provenance, CPU replay; ...
$ git log --oneline -4 -- scripts/benchmark_pairwise_replacement_v2.py
5be34f1 Combined relevance+condition prompt: no gain over the narrow questions; scripts synced
96c974d Correct the reranker controls (left padding), add FollowIR Core17 transfer and the k-pair budget curve
c2e4f84 Reproducibility: ship run scripts, ...
$ git rev-parse 671cbcc:scripts/benchmark_pairwise_replacement_v2.py
fatal: path 'scripts/benchmark_pairwise_replacement_v2.py' exists on disk, but not in '671cbcc'
```
So at the commit that recorded the headline, **the scoring code was not in the repository at all**; it arrived 11h43m later and was then modified for the padding fix and the reranker controls. There is no test that re-runs inference and compares to `exp_direct_probs_dev.json`. What *is* shipped and does check out:

- `scripts/replay_replacement.py --quick` is stronger than a hash check on the JSON in one respect: its ablation path (`replay_ablation` → `choose_pair`, lines 159-176) re-derives the decision itself from the stored two-order probabilities (`exp_direct_probs_dev.json`: `pred.fwd`, `pred.rev`), recomputing normalised A/B distributions, the both-orders `agree` flag, `merged - 0.5 > threshold` and the argmax, and it reproduces the `full` arm (0.364609), `no_abstain` (0.364729), `no_bidir` (0.359637) and `neither` (0.362276) exactly. So `decide()` *is* independently re-checked against raw stored predictions. The `direct`/`contextual` arms take a shorter path (`replay_policy` reads the stored `chosen` flags in `pair_details`, lines 95-124), and — the actual gap — **nothing verifies that the stored probabilities are what the committed harness produces**: there is no inference test, and the harness postdates the artefacts.
- The fixed widened harness agrees with the Direct pairs on the 1,686 overlapping comparisons: mean |Δp| 0.0130, 92.8% within 0.05, **98.9% same side** (`results/replacement_v2/cmp_fixed_vs_direct.txt`). That is decent evidence the Direct probabilities were produced with the correct chat-template + option-name recipe, but it is 1.1% disagreement, not identity — 18/1,686 comparisons flip side.
- `scripts/benchmark_pairwise_replacement_v2.py:150-172` at HEAD does apply the chat template (`apply_chat_template(..., add_generation_prompt=True)`, 4 call sites), left padding (`tokenizer.padding_side = "left"`), `add_special_tokens=False`, and reverses option names symmetrically with the logit indices. The five known past bugs do **not** survive at HEAD.

Fix: route `replay_policy` through `choose_pair` as well (it already exists, so the two arms then have the same independent check as the ablations), add a 20-query inference smoke test that regenerates prompts and asserts probabilities against `exp_direct_probs_dev.json`, and record in `PROVENANCE.json` that the headline artefacts predate the committed harness (produced by the run scripts first committed in `c2e4f84`).

---

## C. Minor findings

### F8 (Minor) — `oracle_decision` is presented as a budget row but is budget-invariant
`expanded3_fixed_policies.json` gives the identical macro 0.36933786238034655 / delta +0.01123933911136931 / counts {41 helpful, 14 harmful, 464 no-swap} at k=2, 6, 12 and 18 — it is computed over the full 18-action pool and ignores k (and coincides with `prior` at k=18, as it must). `REPORT.md:147` prints it as a per-k row `+0.0112 | +0.0112 | +0.0112 | +0.0112`, which reads like a flat curve rather than a constant. Relabel it "oracle over the full 18 actions (not a k-budget row)"; the genuine small-budget ceilings are `oracle_order` (+0.0258 at k=2, +0.0212 at k=6).

### F9 (Minor) — cascade artefact stores a subset macro while the report quotes the full-set macro
`REPORT.md:198-201` says the violation cascade's "macro is exactly Direct's 0.364609", but `results/replacement_v2/exp_violation_cascade_dev.json` stores `macro 0.36038630145994216`, `delta 0.0`, `ci [0.0, 0.0]` — those are 230-query subset values (the "unsure" queries routed to the second judge), not full-set values. A reader diffing artefact against report sees a mismatch. Store both, labelled.

### F10 (Minor, latent) — replay applies swaps by `qid` alone, the exact pattern of the old "chosen not keyed by domain" bug
`scripts/replay_replacement.py`: `out[row["qid"]] = row` (85), `details[(rec["qid"], p["slot_rank"], round(p["delta_ndcg"],9))]` (94), `chosen[r["qid"]] = (p["slot_rank"]-1, p["challenger"])` (101), and the reconstruction loop `for qid, row in cand_by_key.items(): if qid in chosen: ...` (104-109). Harmless on this dev set — all 519 dev qids are globally unique across the 13 domains (checked: 519 qids, 519 unique, 0 collisions) — but on any split or collection where ids repeat across domains it will silently apply a swap to the wrong ranking. Key by `(domain, qid)` and assert uniqueness.

### F11 (Minor) — payload gaps a replicator hits in the first hour
- `configs/replacement_study_20261120.json` names `runs/sub_sft_e2bc_<domain>_dev.trec` (full run); only `..._dev_head10.trec` (top-10) is committed. `scripts/build_replacement_candidates.py` reads the full name, so **the candidate builder cannot be re-run from the repo** even though its output is verified (F-check 3). Either commit the full runs or change the config to the head10 files and document that ranks >10 were never needed.
- `results/ce_ft/TRAINING_CONFIGS.md` states the CE validation ids "cannot be regenerated from the current code" (45 held-out train queries). Disclosed, but it means the frozen baseline's checkpoint selection is not reproducible; say which checkpoint was used and by what criterion.
- `results/replacement_v2/exp_direct_context_dev.json` has no `per_query` block, so the bootstrap can only be replayed by importing `replay_replacement.py`. Add `per_query` (519 floats) to the policy records; it is 4 KB.

### F12 (Minor) — numbers that disagree between documents
- `README.md:75`: "deltas +0.0082/+0.0122/+0.0096/+0.0112, **every CI below 0**" — sign error, all four CIs are **above** 0 (lower bounds +0.0027 / +0.0055 / …). This one is embarrassing because it inverts the sentence's meaning.
- `paper/main.tex:161` prints 873 s for Direct; `REPORT.md` prints 872. Artefact has `cost.seconds` 872.295 and `cost.seconds_total` 872.658, so both roundings come from the same artefact but the two documents disagree. Pick one field.
- `paper/main.tex:218` has doubled signs (`"+-0.0042"`, `"++0.0065"`); `REPORT.md:184` has "32 healthy / 19 harmful" (should be "helpful").
- `paper/main.tex:176` (random budget) is the stale row from F3; `README.md:155-157` repeats it.

---

## D. What survives scrutiny (do not let a reviewer take these from you)

1. **The frozen baseline and its per-domain table are exactly reproducible from raw data.** `results/audit/a1_repro_frozen.py` iterates qrels (so missing queries score 0), binary nDCG@10 with ideal over min(10,|gold|): every one of the 13 per-domain values and both aggregates match (`domain_macro 0.3580985233`, `topic_macro 0.3643666957`). Qrels are binary (all grades 1), so binary nDCG is the right measure; 0 missing queries.
2. **The candidate pool is exactly the documented rule.** `results/audit/a2_pool_rebuild.py` (raw runs only, no qrels): union(dense top-20, CE top-20) − frozen top-10, best-rank order, dense wins ties, docid, cap 3 → `519 queries; frozen_top10 mismatches 0; candidate-list mismatches 0; pool size before cap: min 12 max 29 mean 20.73`.
3. **The 0.3769 ceiling is real.** `results/audit/a3_ceiling.py` (raw runs + qrels, no committed artefact): base 0.3580985233, oracle 0.3769401158, gain +0.0188415926, 68/519 beneficial queries, sum of positive gains 9.276001686. The 0.4098 figure is a different (uncapped) pool and is consistently labelled as such — no stale ceiling.
4. **No gold in the judge path.** Candidate generation is qrels-blind (`grep -c qrels scripts/build_replacement_candidates.py` → 1 hit, a docstring line); the prompt builder reads query text + two passages + the task string; `decide()` reads only `pred.fwd/pred.rev`; `apply_and_score` reads `delta_ndcg` only to report what happened, never to choose. `protocol.labels_used_for_candidates: false` in the headline artefact is accurate.
5. **No train/dev leakage.** dev qids 519; `train_split.json` fold-train 969, overlap 0; `ce_folds.json.fold_of` 969, overlap 0; `results/fusion100_split.json` exclude/eval = 214 train qids, overlap 0. The frozen CE was trained on train groups with `--val-frac 0.12` (45 train queries), so dev was never in any fit.
6. **The Direct−Context null is correctly computed and correctly reported.** Domain-macro paired difference +0.0015 (95% CI [−0.0051,+0.0016]); the paper's p=0.81 matches P(Δ≤0)=0.807. Retained context is not claimed as a contribution.
7. **The hedges are real.** cap2 post-hoc (README:152), 400-word arm and its ns pair (REPORT §9i), FollowIR p-MRR negative (`results/replacement_v2/followir/scores_followir.json` matches README to 4 dp), the withdrawn earlier reranker conclusion, and the explicitly invalidated pre-fix widened run all show a lab that catches its own errors. F1/F3 are failures of bookkeeping, not of honesty.

---

## E. Must fix before submission

1. **Rewrite `REPORT.md` §10 (Conclusion).** Remove "a widened top-100 action space … do not add to that" and the "widened-space failure / AUC 0.532 / +0.0593 / +0.0003" sentence; both are contradicted by §6c/§6d2 and by `expanded3_fixed_dev.json`. (F1)
2. **Commit the widened-action harness and its action cache**, or delete the widened results and the oracle-ordering numbers from the paper and README, since nothing in the repo can regenerate them. (F2)
3. **Fix the `hash()` seed in `scripts/budget_from_cache.py:114`,** regenerate `budget_cache_dev.json`, and replace the random row in `README.md:155-157` and `paper/main.tex:176` with the regenerated values; drop or qualify "the gap-based order does not beat a random pair", which is draw-dependent at k=3. (F3)
4. **Correct the "pre-registered bar" sentence** (`paper/main.tex:173`) or the protocol-file timestamp: the pool/ceiling were frozen at 01:22, the +0.005 bar at 05:11, the result at 02:53. Add a limitations paragraph with the family size (≥120 dev configurations) and the Bonferroni arithmetic — the headline's CI-implied p is 0.0083 and survives m≤6 only. (F4)
5. **State in the abstract/intro that the headline is not the best dev configuration** and why (post-freeze variants, paired intervals against Direct include zero, e.g. 400-word arm +0.0014 [−0.0008,+0.0044]). (F5)
6. **Stop reporting `p = 0.001` as a calibrated p-value**; report the CI plus a permutation p over the 51 non-zero queries, and state that 468/519 per-query deltas are exactly zero. Note the domain-macro vs query-level sign difference for Direct−Context. (F6)
7. **Add `per_query` and a `--from-probs` replay** to the headline artefact, and record in `PROVENANCE.json` that the artefacts predate the committed harness (`c2e4f84`, 11h43m later, then edited twice). (F7, F11)
8. **Fix the stale/sign/wording defects:** "every CI below 0" → "above 0" (README:75); 872 vs 873 s; `+-`/`++`; "32 healthy" → "helpful"; relabel the budget-invariant `oracle_decision` row; label the cascade's subset macro; key replay swaps by `(domain,qid)`; commit the full frozen runs or point the config at the head10 files. (F8–F12)

---

## Appendix — reproduction commands (all read-only except `results/audit/`)

```bash
cd /d/RETECO/github_repo
PYTHONUTF8=1 D:/anaconda/python.exe scripts/replay_replacement.py --quick   # PASS, "matches every committed number"
PYTHONUTF8=1 D:/anaconda/python.exe scripts/check_claims.py                 # PASS, "all claims traceable"
PYTHONUTF8=1 D:/anaconda/python.exe results/audit/a1_repro_frozen.py        # metric from head10 TREC + qrels
PYTHONUTF8=1 D:/anaconda/python.exe results/audit/a2_pool_rebuild.py        # pool rule, 519/519 exact
PYTHONUTF8=1 D:/anaconda/python.exe results/audit/a3_ceiling.py             # 0.3769401158 from raw runs
PYTHONUTF8=1 D:/anaconda/python.exe scripts/budget_from_cache.py --k 1 3 9 --out results/audit/runA.json
PYTHONUTF8=1 D:/anaconda/python.exe scripts/budget_from_cache.py --k 1 3 9 --out results/audit/runB.json   # differs: F3
git log --diff-filter=A --format='%h %ad %s' --date=iso -- scripts/benchmark_pairwise_replacement_v2.py
git log --diff-filter=A --format='%h %ad %s' --date=iso -- configs/replacement_study_20261120.json
grep -rln "oracle_order" --include=*.py .                                   # readers only: F2
```

Files created by this audit (nothing else touched): `results/audit/AUDIT.md`, `a1_repro_frozen.py`, `a2_pool_rebuild.py`, `a3_ceiling.py`, `runA.json`, `runB.json`.
