> **Phase 11 status (2026-11-09, night).** The out-of-fold refit was rerun with the
> correct base model (`bge-reranker-v2-m3`, 567.8 M) after the wrong-base incident,
> for the 1-epoch (`e1b`) and 2-epoch (`e2b`) recipes. Both fits chose RRF dense:CE
> = 1:4 on 45 held-out train queries. Dev: at RRF 1:4, SFT 0.3074 (e1b) / 0.3187
> (e2b) vs zero-shot 0.2981 (+0.0094 and +0.0205, domain CIs above zero); at
> RRF 1:1, e2b reaches **0.3431** vs 0.3166 (+0.0265 [+0.0052,+0.0473]). The
> literal pre-registered bar — SFT at the OOF weight vs the zero-shot best form
> (0.3249) — is not met (e2b 0.3187). The earlier "the frozen 0.3581 does not
> reproduce" reading came from stale `score_sub_sft_e2b_*` files; re-scoring the
> committed runs gives **0.3583**, so the frozen 0.3581 reproduces (further runs
> 0.3575 / 0.3527 / 0.3528, sample sd 0.0029). The SFT arm's advantage therefore
> holds at a fixed form, and the inherited z 1:1 fusion form is *not* the fragile
> part. See `results/PHASE11_VALIDATION.md`.

## 1. Where the system stands

| system | domain-macro | topic-macro |
|---|---|---|
| official BM25 | 0.1147 | — |
| dense mxbai top-100 | 0.2943 | 0.2891 |
| + zero-shot CE, RRF 1:1 (V1) | 0.3166 | 0.3199 |
| + zero-shot CE, z-sum 1:1 | 0.3249 | 0.3260 |
| + SFT CE (1 epoch, lr 1e-5), z-sum 1:1 | **0.3336** | 0.3339 |
| + SFT CE (2 epochs, lr 1.5e-5, best@ep1), z-sum 1:1 | **0.3581** | 0.3644 |
| SFT vs zero-shot at fixed 1:1 (1-epoch arm) | +0.0087 [+0.0037, +0.0142] (domain-clustered) | |
| 2-epoch arm vs zero-shot at fixed 1:1 | +0.0332 [+0.0192, +0.0472] (domain-clustered), 188/78/253 | |
| SFT total vs V1 (1-epoch arm) | +0.0170 [+0.0052, +0.0287] | |

Headroom is in scoring, not recall *in the aggregate*: oracle@10 inside the
frozen top-100 is 0.7446 and the submitted system captures **8.7%** of the gap
(41% of queries improve, 19% worsen). The CE's within-query AUC (0.80) is not
better than the dense retriever's, which is why supervised training — not more
candidate recall — is the only lever that moved the number. Locally, though,
recall is decisive: 27.7% of dev queries score 0 and **36% of those have no gold
doc in the top-100 at all** (`results/DEV_ERROR_ANALYSIS.md`).

## 2. GPT review P0 — status

| # | item | status |
|---|---|---|
| 1 | unify the metric on domain-macro | DONE — `score_run.py`, `aggregate_scores.py`, `bootstrap_compare.py --stat domain`, `fit_fusion_weight.py --select-metric`; refit picks z 1:1 |
| 2 | refit the SFT fusion weight out-of-fold | CLOSED BY DECISION (C′) — the e1/e2 held-out split is not recoverable (replaying it from the current code does not reproduce the logged `docs_loaded` counts, all 13 domains, seeds 0–60), so a 45-query refit would be a 45-query guess (SE ≈ 0.02 ≈ 2× the effect). Instead: keep the inherited 1:1 (fitted on train with the **control** arm, which can only understate SFT), disclose it, and prove weight-robustness with a both-arms dev sweep — `scripts/weight_sweep.py`, `results/WEIGHT_SWEEP.md`. Retrain-with-dumped-split remains available as `scripts/phase11_sft_oof.sh` if a reviewer demands a provable fold |
| 3 | `score_run.py` missing-topic zero-fill | DONE (on GitHub at 6be4e80) |
| 4 | `validate_submission.py --detie` re-validates its output | DONE — old `-1e-9*rank` underflowed on the padding sentinel and left 900 ties/topic; replaced with an ordinal fix |
| 5 | `test_mode_dryrun.sh` overwrite guard | DONE — refuses without FORCE=1 |
| 6 | reproduction material | DONE — `requirements.txt`, `results/MODEL_MANIFEST.json` (sizes+sha256), `results/ce_ft/TRAINING_CONFIGS.md`, `scripts/reproduce_all.sh`, both `train_log.jsonl` |

## 3. What to do next, in order

**P1 — finish the two-week story (no new modelling).**
1. *Out-of-fold fusion weight for SFT.* The 2026-11-09 e1/e2 checkpoints did not
   record which train queries they held out, and the split cannot be replayed
   (documented in `results/ce_val45/val_split_seed17_true.json`). `scripts/phase11_sft_oof.sh`
   therefore retrains the same config with the split dumped by the run itself
   (`--dump-val-qids`), reranks those held-out queries, fits the weight on them,
   and only then reranks dev. Budget: 21 min train + 6 min OOF rerank + ~1 h dev
   rerank, GPU-serial. Expected effect ±0.005 (a weight change cannot move AUC;
   the 8.7%-capture law applies).
   Pass rule: adopt the OOF weight whatever it is; the SFT-vs-zero-shot delta
   must stay >= +0.005 at that same weight. Cost ~0.3 GPU-h.
   *Status 2026-11-10 04:00 — **done; the earlier "does not reproduce" reading
   was a stale-file artefact and is retracted.*** `scripts/phase11_sft_oof.sh`
   retrained e1b (`v2m3_pool8_e1b`) with the correct base model and a dumped
   45-query split; the out-of-fold fit chooses RRF 1:4 in both arms, where the
   2-epoch arm scores 0.3187 (zero-shot 0.2981, +0.0205 [+0.0010,+0.0404]) and
   the 1-epoch arm 0.3074. The "fresh checkpoint scores 0.3196 at z 1:1" number
   came from `results/score_sub_sft_e2b_*_dev.json` files written from
   intermediate fused TRECs; re-scoring the committed runs gives **0.3583**, so
   the frozen **0.3581 reproduces** (seeds 0.3575/0.3528, spread 0.0055). The
   pre-registered literal bar still fails (SFT at its OOF weight 0.3187 vs
   zero-shot best form 0.3249), while the arm-level comparison at a fixed form
   passes: +0.0334 [+0.0191,+0.0477] at z 1:1. Guard:
   `scripts/check_score_freshness.py` (audit after re-score: 0 stale). Details:
   `results/PHASE11_VALIDATION.md`.
2. *Seed replication of the only significant gain.* Seeds 23 and 29 for the
   **reproducible 2-epoch recipe** (`v2m3_pool8_e2b`: 2 epochs, lr 1.5e-5,
   8 frozen layers), each followed by a 13-domain dev rerank and the frozen
   1:1 arm (`scripts/phase11_seed_repl.sh`, tags `v2m3_s23_e2` /
   `v2m3_s29_e2`). Keep SFT in the title iff the mean delta >= +0.005, at least
   2/3 seeds positive, and the pooled domain-clustered CI excludes 0.
   Status 2026-11-10 03:49 — **DONE.** Seeds 23 and 29 trained and reranked on
   all 13 domains (`v2m3_s23_e2`, `v2m3_s29_e2`). Dev z 1:1: **0.3575** (seed 23)
   and **0.3528** (seed 29) vs zero-shot 0.3249; bootstraps +0.0325
   [+0.0184,+0.0470] and +0.0279 [+0.0160,+0.0405]. Mean of the three fresh
   runs 0.3562, spread 0.0055, all three positive against zero-shot ⇒ keep SFT
   in the title.
3. *Synthetic test split replay.* Copy dev, rewrite ids, run dense -> CE ->
   fusion -> official `format_checker` end to end. Prevents a submission-day
   surprise; produces the exact command for January. Cost 0 GPU-h (~1 h).
   Status 2026-11-10 04:31 — **DONE on GPU** (`scripts/phase13_test_replay.sh law
   quant iota`, 24 synthetic test topics, frozen `v2m3_pool8_e2` + z 1:1).
   Official `format_checker.py`: 0 errors, VALID on all three; strict-run sanity
   now passes too (PHASE 13: PASS). The replay caught a real submission-day bug:
   two adjacent rank pairs in `quant/70097_1088` printed identical scores because
   the de-tie compared raw floats (a 1e-9 gap rounds to the same six decimals)
   and the tail filler `-1e9 + 1e-6*i` made the de-tie step 1000 per row. Both
   fixed in `scripts/make_submission.py` (printed-space de-tie by 1e-6;
   rank-spaced filler `-1e9 - 1e-3*i`). Details: `docs/SUBMISSION.md`.
4. *Report the robust views as primary alongside the headline:* 12-domain
   (SFT 0.3181 vs zero-shot 0.2832) and n>=19 (0.3267 vs 0.2985), measured on
   the fresh 2-epoch checkpoint at z 1:1 — the headline's 0.3581 leans on iota,
   which has only 3 dev queries. Status 2026-11-10: **done**, numbers in
   `results/PHASE11_VALIDATION.md` and the paper's robustness paragraph.

**P2 — evidence for the paper (CPU-only, highest value per hour).**
5. *Annotate year-mismatched gold documents* — **DONE** (2026-11-09):
   `results/MISMATCH_ANNOTATION.md`, 35 def-2 documents, 20 double-coded,
   κ = 0.84; retrospective 13, incidental_year 11, continuing 8, period_shift 2,
   wrong_time 1 — only 1/35 is genuinely about another period, which turns
   "mismatch is not irrelevance" from an inference into a table.
6. *Temporal-ceiling analysis on the submitted mxbai top-100* — **DONE**:
   `results/TEMPORAL_CEILING.md` now uses the deployed mxbai candidates, the
   6/8 vs 7/8 miscount is fixed (oracle lowers nDCG in 7 of 8 domains), and a
   provenance section records the correction.

**Kill list (do not spend time on):** per-domain fusion weights (fitted on
train: negative), the 1.5B reranker (OOM risk on 8 GB, expected +~0.002), more
hand-built temporal features (ten families already null), any document-ID
feature (leaks the retrieval pipeline).

## 4. Paper in one line

"Temporal consistency is not the binding constraint in RETECO Track 1a: text-
extractable time evidence is too sparse and time mismatch is not irrelevance, so
a temporal-agnostic two-view retriever-reranker with train-fitted fusion is the
strongest 8 GB-viable system, and the remaining headroom is scorer
discrimination, not temporal modelling."

Sections: intro/thesis -> benchmark anatomy (observability) -> system ladder ->
fusion protocol -> supervised reranking -> temporal null + ceiling -> why
nothing else moves -> limitations/reproducibility. Claims to cut or requalify:
"ten pre-registered negatives" (only the gate has a frozen protocol), "temporal
signals do not work" (only no gain under our protocol), "fusion is better" (CI
crosses zero), "SFT validated" (one seed).

## 5. Update 2026-11-09 ~14:40 (afternoon)

- **A second SFT checkpoint (2 epochs, lr 1.5e-5, `v2m3_pool8_e2`) is better than
  the frozen one.** Val was inconclusive (0.3029 vs 0.3008), but the first four
  completed dev domains give e2 − e1 = +0.0058 bitcoin, +0.0211 cardano,
  +0.0355 economics, +0.0222 genealogy (105 queries, 34 up / 14 down). Full
  13-domain rerank is running (job bash-1236, tag `sft_e2bc`); if the aggregate
  holds, e2 replaces e1 as the frozen submission arm — freeze with
  `TAG=sft_e2bc bash scripts/freeze_sft_arm.sh` (fusion → de-tie+validate →
  score → aggregate in one command).
- **Depth analysis** (`results/DEPTH_ANALYSIS.md`): dense Recall@300 = 0.8183
  (0.6995 at 100) and the pool's oracle nDCG@10 rises 0.7446 → 0.8513, i.e. 22.5 %
  of gold docs sit at ranks 100–1000. A 62-query probe (`scripts/run_depth_probe.sh`)
  decides whether the CE's 8.7 % capture survives at depth; the decision rule
  (+0.01 on the probe, CI excluding 0) is fixed in the doc before the run.
- **Paper draft** now at `docs/PAPER_DRAFT.md` (447 lines, every number carries
  its source file in an HTML comment); a verification pass against the artifacts
  is in flight.
- Submission path is test-phase ready: `scripts/make_submission.py` passes
  `--examples` to both stages, keys the CE cache by checkpoint, de-ties inside
  the writer; runbook in `docs/SUBMISSION.md`.
- **Artifact bug found and fixed** (paper-verification pass): the first
  `runs/sub_zs11_*` freeze was built from `runs/fused_1_1_*`, which are the *RRF*
  V1 runs (`tag=rrf`, score 0.0317) — i.e. it was a byte-level duplicate of V1.
  The real zero-shot z 1:1 runs are `results/ss_dev/ss_1_1_<d>_dev.trec`; they are
  now de-tied/validated into `runs/sub_zs11_*` (13 files, 0 errors,
  `results/submission_validation_zs11.json`) and rescored to
  `results/aggregate_sub_zs11_dev.json` = **0.3249 domain-macro / 0.3260
  topic-macro**, matching the number in Table 2. `RUN_MANIFEST.sha256` now covers
  all four frozen sets (52 files: sub_1_1, sub_1_2, sub_sft11, sub_zs11).
- **The 2-epoch checkpoint is a large win — provisional new best.** Full 13-domain
  dev rerank (`TAG=sft_e2bc`, freezing via `scripts/freeze_sft_arm.sh`) gives
  **0.3581 domain-macro / 0.3644 topic-macro** (`results/aggregate_sub_sft_e2bc_dev.json`,
  13 files, 0 validation errors). Against the frozen zero-shot z 1:1 baseline:
  **+0.0332**, query CI [+0.0161,+0.0523], domain-clustered [+0.0192,+0.0472],
  188 better / 78 worse / 253 tied (`results/boot_dom_e2_vs_zs.json`). Per domain
  (`results/WEIGHT_SWEEP_E2.md`): hsm +0.071, law +0.076, workplace +0.063,
  cardano +0.055, history +0.045, economics +0.039; only bitcoin −0.002 and
  quant −0.001 fall. The gain is flat in the fusion weight (+0.009 at w=0.5,
  +0.025…+0.035 for w≥0.85, 13/13 domains up at w=0.7/1.5/1.8), so it is a CE
  effect, not a re-tuning of the fusion. The chosen checkpoint is epoch 1 of the
  2-epoch run (val 0.3029); epoch 2 was worse (0.3012), so the effective change
  vs e1 is peak lr 1.5e-5 with a 2-epoch (slower) decay — **not** more epochs.
  Replication with seed 23 is training (`v2m3_pool8_e2s23`); adopt e2 in the
  paper only if it replicates.
- **Depth probe: negative** (`results/DEPTH_ANALYSIS.md`). Reranking the 200-deep
  pool on the 62 hardest queries changes nDCG@10 by +0.0010 [−0.0009,+0.0036]
  (zero in 9/13 domains): the CE scores deep candidates near the floor, so a
  larger pool buys nothing until the CE is trained on deeper negatives. The k=300
  full rerank is cancelled; the "headroom is ranking" claim holds.

## 6. 工程守则（2026-11-09 血的教训）

- **8 GB GPU 上任何加载模型的任务都不能与 CE 训练并行**；并行会 OOM 打死训练（已发生 3 次）。
- 长训练用 `ops/launch_train29.ps1` 这类 `Start-Process -WindowStyle Hidden` 脱离宿主作业树启动，
  监视只看 `train_log.jsonl`，**不要**用 `best/` 目录作为“训练结束”的信号（`best/` 在第 1 个
  epoch 末尾就会出现）。
- `scripts/phase11_*.sh` 已移到 `ops/_disabled_20261109/`：它会启动错误配置的 e1b 训练，
  曾两次抢 GPU。不要再运行。

## 7. 种子复核完成（2026-11-09 20:30）

- seed 29 全流程跑完：dev 域宏 **0.3527**，对 zero-shot **+0.0278**
  （查询 CI [+0.0118,+0.0455]，域聚类 [+0.0159,+0.0404]），12 域 0.3123，n>=19 0.3232。
- 两个种子都为正且 CI 排除 0，协议门槛通过；主结论统一为 **+0.031 ± 0.003 over zero-shot**，
  冻结提交仍是 seed 17 的 0.3581。详见 `results/REPLICATION.md`。

## 8. Update 2026-11-12 15:30 — Phase 17 结账（dev 端全部落定）

**一句话**：五个 dev 级候选全部为负（−0.011 ~ −0.014），冻结件 0.3581 保留；
提交件不变，唯一变化是论文与日志现在把"重训带 0.347–0.358"写明。

| 臂 | val（45 query） | dev domain-macro | Δ vs 冻结 0.3581 |
|---|---|---|---|
| 冻结 sft_e2bc（v2m3_pool8_e2/**ep1**） | 0.3029† | 0.3581 | — |
| `lx_possum`（同一 loss 重训，新代码路） | 0.2865 | **0.3470** | −0.0111 |
| `cx_qv`（64 词全问题） | 0.2920 | 0.3443 | −0.0138 |
| `cx_bf`（补 140 无正例 query） | 0.2533 | 0.3443 | −0.0138 |
| `sx_l03` / `sx_l10`（分步监督） | — / 0.2913 | 0.3442 / 0.3438 | −0.0139 / −0.0143 |
| 种子 23 / 29（旧路径重训） | — | 0.3575 / 0.3528 / 0.3527 | +0.0 / −0.005 |

† 冻结臂的 val 是旧划分/旧负例池上的 plain nDCG@10（ep1 0.3029 vs ep2 0.3012，`best` 取 ep1，md5 ff2b4ab8 = `v2m3_pool8_e2/epoch1`），与 `lx_*` 系列的 0.2865（domain-macro、新池）**不可比**；`lx_possum` 的 `best` 是自己的 ep2（0.2865 > 0.2790 > 0.2730）。

**口径教训（本轮最重要）**：`cx_bf` 的 topic-macro 是 0.3711，**高于**冻结的 0.3644——
同一份 run，按旧口径是 +0.0067 的胜利，按官方 domain-macro 是 −0.0138 的失败。
论文/日志/汇报一律用 domain-macro，且必须注明口径。

**工程事故（详见 docs/RESEARCH_LOG.md §5p.5）**
- 02:18 系统睡眠 7 秒 → CUDA 上下文被销毁、进程空转 11 小时。GPU 长跑必须配
  `ops/keep_awake.ps1`，验证只看 `powercfg /requests` 的 SYSTEM 段。
- PowerShell 里 `0x80000000` 是负 Int32，`SetThreadExecutionState` 必须传
  `[uint32]2147483648` / `[uint32]2147483649`。

**还剩的活（2026-11-12 19:40 已全部关闭）**
1. ✅ CLS 池化基线跑完：`results/aggregate_dense_mxbai_cls_dev.json`，13 域 domain-macro
   **0.2956** vs mean 0.2943（+0.0014）、topic-macro 0.2827 vs 0.2891（反向 −0.0064）→ wash，
   dense 基线不改；hsm 曾因一条 44.5 M 字符的文档触发 tokenizer OOM，已按字符预截断修复
   （`scripts/run_dense_generic.py`），hsm 重跑 54 min 补齐。
2. ✅ 写进 `docs/RESEARCH_LOG.md` §5p.6；论文 §3 加 "0.2956 with CLS pooling: a wash"、
   §5.7 结尾改成"五步重训皆输 + 第六步是 wash"、§7 的池化消融措辞收窄到 CE 窗口池化。
3. ✅ `docs/RESEARCH_NARRATIVE.md` §14 表已更新；提交见 github_repo commit（2026-11-12 第二轮）。

## 9. External audit response (2026-11-13)

- 审计钉在 `681cd80`，四类问题逐条核实；仓库缺的 `ops/`、全部 `scripts/`、65 份逐查询评分、
  100 深 rerank run、`val_qids_e2.json`、训练组与 step targets 已全部提交（`1059995`）。
- 归因拆分：`lx_possum` 0.3470（−0.0111 同代码重训成本）；四臂相对它 −0.0026/−0.0026/−0.0027/−0.0031，
  域 CI 全跨零（`results/boot_dom_*_vs_lxpossum.json`、`ATTRIBUTION_LXPOSSUM.json`）。论文 §5.5/§5.7 已重写。
- 深度修正：96.3% 是 depth-1000；CE 只看 100，对应 80.7%/88.3%/51.4%；138 个零分 query 中 52 个 top-100 无相关。
- 年份规则改称"固定重排规则失效"，不再叫天花板/上界；训练组是 top-100 里抽的 20 篇（1,071 组/21,420 对）。
- 代码加固：模型目录**内容指纹**进缓存 key；无 sidecar 的旧缓存默认拒绝（`--adopt-legacy` 才收养）；
  `validate_submission.py` 无法解析域名时报错，缺 qrels/corpus 报错。
- 剩余：fused 1000 深 run 未提交（每臂 37 MB，可由 dense top-1000 + CE 分数重建）；官方 test query 仍未放出。
