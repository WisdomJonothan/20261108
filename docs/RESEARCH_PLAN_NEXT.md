# RETECO Track 1a — 下一阶段实验计划（v1，2026-11-10）

本文件是 11-09 收尾之后的实验路线：从一个"已冻结、可复现、能提交"的系统出发，
按用户 11-10 备忘的六个阶梯做增量改进，并检验两条创新主线是否真的成立。

**当前基线（不要动）**：域宏 **0.3581**（`results/aggregate_sub_sft_e2bc_dev.json`），
+0.0332（域聚类 CI [+0.0192,+0.0472]）；同配方另外两种子 0.3575 / 0.3528；
平均 0.3561 ± 0.0029。复现口径见 `results/REPLICATION.md`。

---

## 0. 诊断（备忘 §1 的代码事实，逐条在源码里核对过）

| # | 现状（`scripts/train_ce.py`） | 后果 |
|---|---|---|
| 1 | 选 checkpoint 用 `np.mean(all_scores)`（全验证查询直接平均） | 大域（history 526/1071≈49.1%）主导选择；iota 只有 7 条 |
| 2 | 损失 `logsumexp(all) - logsumexp(pos)` = `-log Σ_{P} p(d)` | 一篇容易正例吃满即可低损失，多正例不额外要求 |
| 3 | 训练丢掉"Top-100 无正例"的 140 条 query（1211→1071） | 训练分布与评测分布不一致，且丢掉最难的样本 |
| 4 | query 只用标题（`clean(v.split("<p>")[0])`） | 时间范围 / 排除 / 比较条件被删掉 |

损失问题已量化（同一份 12 候选、3 正 9 负）：

| 候选排序 | possum（现状） | allpos | ndcg |
|---|---|---|---|
| 一篇正例吃满、两篇埋在负例下 | **0.0541** | 3.1083 | 0.1958 |
| 三篇正例全在前排 | **0.0511** | 1.4587 | 0.0181 |

possum 几乎不区分这两种排序 → 备忘的诊断成立。

---

## 1. 已落地的代码设施（默认行为不变，全部 opt-in）

`scripts/train_ce.py`：

| 参数 | 取值 | 作用 |
|---|---|---|
| `--loss` | `possum` / `allpos` / `ndcg` | 现状 / `-(1/|P|)Σ log p(d)` / LambdaRank 按 |ΔNDCG@10| 加权的交换梯度 |
| `--val-metric` | `mean` / `domain_macro` | checkpoint 选择信号（`--val-metric domain_macro` 用域宏） |
| `--sampler` | `uniform` / `sqrt` | 每轮按 `P(域)∝√n_域` 抽样（小域抬升但不重复到等量） |
| `--step-lambda` + `--step-targets` | 浮点 / 目录 | 叠加官方分步证据项 `-(1/m)Σ_j log Σ_{d∈P_j} p(d)` |
| `--q-mode` | `title` / `body` / `full` / `ctxt` / `summ` | 查询表示四选一，共享 `--q-cap-words` 词预算 |
| `--eval-qvariants` | `title,body,full,ctxt,summ` | 在验证集上只评一次各表示然后退出（不训练） |
| `--dump-val-qids` | 路径 | 固化一次切分，后续所有实验共用（`--val-qids` 复用） |
| `--init-from` | 目录 | 从已训 checkpoint 起步（配合 `--eval-qvariants` 做只评不训） |
| `--report-steps` | 开关 | 验证时顺带记 `step_hit_rate` / `pos_in_top10` / `q_with_2pos_top10` |

`scripts/build_step_targets.py`（新）：把 `data/track1_tempo/<d>/steps_train.jsonl`
+`qrels_steps_train.txt` 转成 `results/ce_step_targets_<domain>.json`
（`{qid: [[step1 gold docs], [step2 gold docs]]}`），13 个域全有。

`scripts/build_ce_train_groups.py`（改）：新增 `--backfill-gold N` —— 对首阶段没召回到
任何正例的 query，把官方标签里的真实正例直接注入组内（评测时仍不可见）。

缓存指纹（备忘 §3 第一条，本轮落地）：`run_dense_generic.py` 的 doc_emb 缓存旁写
`*.fp.json`（model / pooling / max_length / dtype / prefix / 文档文本 sha1），指纹不符
直接 SystemExit，旧缓存只警告照用，`--refresh` 强制重算；`make_submission.py` 的
dense/CE run 同样带指纹（查询文件 sha1、索引目录、模型、pooling、prefix、topk；CE 再加
dense run 的 sha1 与 q-mode/q-cap/win/pool/k），融合配置与两个指纹写进 `<out>.config.json`。
`--strict-cache` 连“没有指纹的旧缓存”也拒绝；RRF 分支的权重改为 `--rrf-wa/--rrf-wb`。

---

## 2. 数据事实（本轮实测）

- 分步标注：13 域 **1,121 query / 2,762 step**。其中 **2,321/2,547 = 91.1%** 的步骤在现有
  训练组里至少有一篇正例文档 → 步骤监督不必等候选扩展就能生效；剩下 8.9% 就是要补的漏召回。
- 补回漏召回：原 1,071 query / 2,876 正例 → **1,211 query / 3,170 正例**（+140 query，+294 正例；
  负例 18,544→21,050）。逐域丢弃数：history 35、genealogy 19、bitcoin 17、politics 15、
  hsm 13、economics 10、travel 10、monero 8、cardano 5、quant 5、workplace 3、iota 0、law 0。
  产物 `results/ce_train_groups_bf_<d>.jsonl` + `_bf_meta.json`（`--groups-prefix` 指向它即可）。

---

## 2b. 约束丢失的定量证据（新）

扫 13 域 train+dev 共 **1,730 条 query**（`results/QUERY_CONSTRAINT_STATS.json`）。
标题平均 12.1 词、正文 151 词。每类线索在正文出现 / 在标题出现 / 只在正文（被“只用标题”丢掉）：

| 线索 | 正文 | 标题 | 仅正文 | 占全部 query |
|---|---|---|---|---|
| 年份 | 363 | 184 | **247** | 14.3% |
| 时间关系（after/before/between…） | 440 | 223 | **328** | 19.0% |
| 排除（not/except/without…） | 304 | 70 | **282** | 16.3% |
| 比较（more than/less than/first…） | 310 | 148 | 243 | 14.0% |

**至少一类约束线索只在正文、被标题化丢弃：775 条 = 44.8%**。逐域条数：history 343、
hsm 77、politics 71、genealogy 59、travel 52、bitcoin 38、economics 35、monero 34、
cardano 18、workplace 17、quant 16、law 12、iota 3。dev 上 241/519 = 46.4%，够做子集检验。

定义以代码为准：`scripts/build_query_constraint_stats.py`（正则写在文件头），同时产出
`results/QUERY_CONSTRAINT_FLAGS.json`（逐 query 的 `cues_lost` / `any_lost`）。
**更正**：早先版本报过 666/907/826/659 与 56.1%，但那个内联统计没有留脚本、无法复现；
现按可复跑脚本重算，量级与结论不变（"近半数 query 的约束只在正文里"），数字以本表为准。

---

## 3. 实验阶梯与状态

| 步 | 内容 | 状态 |
|---|---|---|
| 0 | 统一配置 / 缓存 / 预处理 / 评分口径 | ✅ 已固化（`--dump-val-qids` + 同一 `--win/--stride/--max-len/--max-windows` + 两条缓存指纹链） |
| 1 | 域平衡采样 + 域宏选 checkpoint | ✅ 已完成：四臂同一 45 条 val，`lx_sqrt` 0.2912 最高但只比 `lx_possum` 0.2865 高 0.0047 < 0.010 并列带 → 保留当前损失；域宏与 plain mean 在四个臂上选出同一 epoch（都是 e2） |
| 2 | 三种损失对照 | ✅ 已完成：possum 0.2865 / allpos 0.2809 / ndcg 0.2764（Δ vs possum −0.0055 / −0.0101，配对 CI 均含 0；零分 query 数两臂相同 16/45） |
| 3 | 标题 vs 同词预算的约束保真短查询 | ❌ **已判负（2026-11-12）**：dev `cx_qv` 0.3443（−0.0138，子集检验见 §5p.1）；val 五臂全部并列（full 0.2869 / title 0.2827 / ctxt 0.2744 / body 0.2703 / summ 0.2614，均 <0.010 带） |
| 4 | 在最佳普通排序基线上叠加步骤监督 | ❌ **已判负**：`sx_l03` (λ=0.3) dev 0.3442、`sx_l10` (λ=1.0) dev 0.3438，对冻结 0.3581 Δ −0.0139 / −0.0143（域 CI 均含 0），步骤覆盖率反而 59.45%→55.6%/56.0%；而 45 条 val 上 λ=1.0 是六臂第一 → 训练内划分选错臂。见 `results/STEP_ARM_sx_l0{3,10}.md` 与 `docs/PAPER_4PAGE.md` §5.6 |
| 5 | 补回漏召回正例 + 深层困难候选 | ❌ **已判负（2026-11-12）**：`cx_bf`（1,211 query / 294 额外正例）dev 0.3443（−0.0138）；按 topic-macro 反而是 +0.0067，口径必须写 domain-macro（§5p.2） |
| 6 | 固定预算多来源候选 + 最终融合训练 | 🚫 **未做**：4/5 判负后失去前提。剩余等价工作降级为 CLS 池化基线（2026-11-12 完成）：13 域 domain-macro 0.2956 vs mean 0.2943（+0.0014）、topic-macro 方向相反，判为 **wash**，dense 基线不改（`results/aggregate_dense_mxbai_cls_dev.json`，见 RESEARCH_LOG §5p.6） |

队列 `ops/queue_loss_exp.sh`（日志 `logs/loss_exp.log`）先等 GPU 空闲，随后
**共用同一条 seed-17 切分**（`results/val_qids_e2.json`）跑四臂：

```bash
--epochs 2 --lr 1.5e-5 --warmup-frac 0.08 --freeze-layers 8 --window-mode max \
--max-windows 3 --max-len 320 --pool-chunk 32 --val-frac 0.12 --val-queries 4 \
--val-depth 100 --seed 17 --val-metric domain_macro
```

| 臂 | 变量 |
|---|---|
| `lx_possum` | 现状损失（只把选模型换成域宏）→ 隔离"选模型"的影响 |
| `lx_allpos` | 多正例损失 |
| `lx_ndcg` | nDCG@10 导向 |
| `lx_sqrt` | 现状损失 + √n 域平衡采样 |

每臂日志打印 epoch0（零样本参考）/e1/e2 的 mean 与 domain_macro，`best` 按域宏选。

---

## 4. 两条创新主线（备忘 §2）

- **A（主线）多证据排序监督**：`L_step = -(1/m) Σ_j log Σ_{d∈P_j} p(d)`，用官方步骤-文档标注
  做训练信号，测试时仍只吃"完整问题 + 文档"。可行性 = 上面 91.1% 的步骤覆盖率。
  应报告：步骤级命中率、多正例进前十比例、以及它是否真的提高域宏（而不是只提高"容易的那篇"）。
- **B 约束保真查询表示**：`ctxt` 表示（标题 + 带约束线索的正文句，同词预算）。
  需要证明它不是重复 Stage-Aware Decomposition（后者只在重排阶段用子问题）。
- **C 直接训练融合**：`S = b(q,d) + λ·r_θ(q,d)`，让 CE 学补 dense 的错；另需澄清
  "z 归一化抹平尺度"的说法（z 对正比例缩放族不变，真正要查的是相对分差、排序互补、极值、缓存）。

### 4b. 预登记的成功判据（写于实验之前，2026-11-10 05:40）

| 创新 | 主判据（dev，519 query） | 机制判据（必须在同一批 query 上同向） | 反例（判负） |
|---|---|---|---|
| A 步骤监督 | 相对"最好的普通排序臂"域宏 `+0.005` 且域聚类 95% CI 不跨 0 | `step_cov@10` 与 `pos_in_top10` 同时上升；`multi_at_10` ≥ +2pp | 域宏不升、只有 `multi_at_10` 升（说明只是把第二篇拉近，没解决"只找最容易那篇"） |
| B 约束保真 | 同预算 `ctxt` 优于 `title`/`body`/`summ` 中的最好者 `+0.005` | 在被标题化丢掉线索的 44.8%（dev 46.4%）子集上增益显著大于其余 | 增益集中在无约束 query（说明只是"多加词"的副作用） |
| C 融合训练 | 相对 z 1:1 冻结形式 `+0.005` 且 CI 不跨 0 | 排序互补度（dense 与 CE 的 top-10 Spearman）在增益最大的域上最低 | 只在 dense 已经很好的域上提升（等于重复 dense） |

任何一项只满足主判据而机制判据反向，按"未证实"处理，不写进贡献列表。
B 的子集检验：`scripts/eval_constraint_subset.py` 按 `results/QUERY_CONSTRAINT_FLAGS.json` 的逐 query 标记切分（已实现，冻结系统的结果见 `results/CONSTRAINT_SUBSET.md`）。A 的机制判据由 `scripts/dev_multievidence.py` 在 dev run 上算。A 的主判据基线取 `sft_e2bc`（0.3581）：`lx_possum` 与它只差 checkpoint 选择口径，步骤臂的 dev 数字若晚于基线可用，也可用 `lx_possum` 的 dev 值替换。

---

## 5. 报告规范（备忘 §4，全部实验适用）

1. 只在 **Train 内**按 query 分组开发；负例挖掘必须在划分之后做；dev 只在最后确认时碰一次。
2. 主指标 = 官方域宏（先按 query、域内平均、13 域等权）；同时给每域增减、配对 bootstrap CI、
   多种子波动。
3. 解释性指标：时间覆盖率、多正例进前十比例、漏召回恢复率。
4. 不再用"AUC 必须 +0.01"这类单一门槛做淘汰。
5. 工程：`make_submission.py` 的 RRF 分支用常量权重（`--wa/--wb` 只作用于 z 融合），
   缓存必须绑完整查询文件 + 参数指纹，评测/复现/提交共用同一配置。
---

## 5b. 内部划分的分辨率（实测，2026-11-10）

45 条训练内 query（13 域各 4，iota 1/law 3/quant 2/workplace 3）的配对域聚类 bootstrap：
`lx_possum` epoch2 vs 零样本 = 0.2865 vs 0.2730，**Δ+0.0135，CI [−0.0317,+0.0571]，P=0.727**
（`results/VAL_BOOT_lx_possum_e2_vs_zs.json`）。即内部划分的分辨率约 **±0.03–0.04**，远大于
我们关心的 +0.01 级差异。据此：val 只做粗筛（并列带宽 0.010，并列偏向更简单臂），
胜负由 dev 519 条的配对 bootstrap 决定；`ops/final_eval.sh` 至少跑冠军与亚军。

## 6. 运行记录（2026-11-10 05:34 起）

### 6.1 已验证的流水线（可复跑）

| 脚本 | 作用 | 已验证点 |
|---|---|---|
| `ops/queue_loss_exp.sh` | 四臂损失/采样对照（`lx_possum/lx_allpos/lx_ndcg/lx_sqrt`） | `bash -n`；`--dump-val-qids` 冒烟通过；05:34 起 `lx_possum` 训练中（942 组 / 45 val） |
| `ops/queue_stage2.sh` | 胜者 + 五种查询表示（`title/body/full/ctxt/summ`，同 64 词预算） | 等 `=== loss experiment done`；输出 `results/LOSS_WINNER.json` → `results/QVARIANT_WINNER.json` → `results/LOSS_ARMS.md` |
| `ops/queue_step_exp.sh` | 步骤监督 `sx_l03`(λ=0.3) / `sx_l10`(λ=1.0) | 等 `QVARIANT_WINNER.json`（90 分钟超时兜底） |
| `ops/queue_stage3.sh` | 约束保真 `cx_qv` + 补全组 `cx_bf` | 等 `=== step experiment done`；+0.002 才换胜者 |
| `ops/final_eval.sh` | 对最终胜者一次性 dev 重排 → 融合 → 校验 → 评分 → bootstrap | 五个下游脚本 `--help` 逐个核对过参数名 |
| `scripts/compare_val_arms.py` | 两臂在**同一批 val query** 上配对 bootstrap（query + 域聚类） | 需要两臂都带 `--dump-val-scores` |
| `scripts/summarize_arms.py` | 各 epoch 的 mean / 域宏 / best / `ep@mean` / `dm@mean` | 顺带回答"选择指标是否改变选点"，不需额外训练 |

### 6.2 判定规则（写死在队列里）

1. `best` 一律按 **val 域宏**，不再按 plain mean。
2. **val 只做粗筛，dev 定胜负**。val 只有 45 条 query（13 域），单条 query 的 nDCG 方差
   放大 3.4 倍（相对 519 条的 dev），经验噪声约 ±0.01——所以 val 上 `Δ < 0.01` 的一律视为并列，
   并列时偏向更简单/更接近现状的臂；真正的胜负看 dev 上 519 条的配对 bootstrap（query + 域聚类 CI
   不跨 0），只报 Δ 不报 CI 的结果不许进论文。
3. 因此 `final_eval.sh` 至少跑**冠军 + 亚军**（必要时加现状臂），每臂约 34 GPU-min + 评分 2 min。
4. 同一批 val query（`results/val_qids_e2.json`，seed 17，按 `--val-frac 0.12` 抽签固化）。
5. 查询表示与损失共享同一 `--q-cap-words`、`--max-len`、`--max-windows`，避免"表示赢在预算上"。

### 6.3 已知的坑

- 本机 `CUDA_VISIBLE_DEVICES=` 空值**不能**禁用 CUDA；`= -1` 才有效，且真 CPU 训练会 segfault（rc=139）。
- WDDM 下 `nvidia-smi` 常驻约 3 GB，内存阈值判断不了空闲；改用 `ops/gpu_busy.ps1` 扫 python 命令行。
- 用 heredoc 往 `.sh` 里写 `\`+换行会被吞成空格（命令仍合法，但续行没了）；heredoc 里的 `\x00` 会变成真 NUL 字节。
- `results/ce_ft/qvar_eval/` 是早期被杀的 qvariants 残留，可删。
