# RETECO Track 1a — 研究日志（长期参考版）

> 目的：把每一次实验的**问题、假设、设计、结果、解释、决策、错误**写清楚，
> 让任何人（包括未来的我）不用读聊天记录也能复现和继续。
> 所有数字都是本机真实运行的结果；引用官方数字时会明确标注。
> 最近更新：Phase 2 进行中（2026-11-07）。

---

## 0. 项目与约束

- 任务：SemEval-2027 RETECO，Track 1a（Temporal Retrieval）。输入 query + 所属领域语料，输出文档排序；官方指标 nDCG@10。
- 首轮范围：IOTA（9,680 文档 / 10 条 query）、Bitcoin（76,706 / 100 条 query）。Phase 2 加入 Economics（74,692 / 83 条 query）。
- 硬约束：不得用文档 ID 前缀/数字大小/gold 命名规律；不得让测试标签进入检索；不得拿 dev 反复调参后当独立验证；论文里不得宣称"每个 query 都在 116 万文档中检索"（每个领域独立语料）。
- 环境：`D:\anaconda\python.exe`（3.12.7）、JDK 21（`D:\RETECO\env\jdk-21.0.13+9`）、RTX 4060 Laptop 8GB。
- 目录：`data\`（官方数据）、`indexes\`（Lucene 索引）、`runs\`（TREC run）、`results\`（评分/meta/诊断）、`logs\`、`scripts\`、`emb\`（向量缓存）、`dist\`（结果包）。

## 1. 研究问题与当前答案

| # | 问题 | 当前答案 | 证据 |
|---|---|---|---|
| Q1 | 官方评分链路能否在本机复现 | 能，四个公开基线逐位复现 | 0.3309/0.1563/0.0370/0.0856 |
| Q2 | 官方预建索引能不能复现官方基线 | 不能（实现不同，top-10 重合 ~60%） | Phase 0 §3 |
| Q3 | 零样本 dense 与 BM25 谁强 | dense 显著更强（Bitcoin dev：dense 0.2171 vs 官方复现 BM25 0.0370；本地预建索引 BM25 0.0300） | Phase 0 §4 |
| Q4 | 召回还是排序是瓶颈 | 排序（dense top-100 理想 0.6488 vs 实际 0.2171） | Phase 0/1 |
| Q5 | 时间信息能不能用元数据 | 不能，语料只有 id+content；gold 仅 12% 含年份 | Phase 0 §5 |
| Q6 | 通用 cross-encoder 能不能超过 dense | 强模型追平（0.2121 vs 0.2171，n.s.），弱模型显著变差 | Phase 1 |
| Q7 | chunking/截断是不是瓶颈 | 不是：chunk 无收益；gold 96.4% token 可见 | Phase 1 |
| Q8 | 融合有没有用 | 有正收益但不显著（pooled +0.0156，CI [−0.006,+0.038]） | Phase 1 §3 |
| Q9 | 结论能否跨领域 | **Phase 2 进行中** | — |
| Q10 | listwise/LLM 精排能否造出强中间层 | **Phase 2 进行中** | — |

## 2. 实验规范（每次实验必须满足）

1. **冻结候选**：同一候选集下比较不同排序器，否则测的是召回不是排序。
2. **深度一致**：任何 oracle/Recall 的数字必须与候选深度绑定，字段名带深度（`oracle_nDCG@10_cand100`）。
3. **一致性断言**：候选里没有 gold 时 oracle 必须为 0，脚本内断言，全量扫描零违例。
4. **train 开发、dev 确认**：配置选择只在 train 上做，dev 只在最后跑一次。
5. **配对 bootstrap**：任何"提升"都要给 95% CI 与 P(Δ>0)，不只看均值。
6. **官方 scorer 交叉核验**：自研评分脚本必须与 `scorer.py` 一致（`PYTHONUTF8=1`）。
7. **不做 ID 特征**：检索/重排代码不读 qrels（已 grep 验证）。
8. **失败也记录**：每个跑过的配置都进表，包括变差的。

## 3. Phase 0：官方链路与基线

**设计**：先复现官方 BM25（starter-kit 的 gensim 实现）→ 再用官方预建 Lucene 索引跑同一 BM25 → 零样本 BGE dense → 计算 Recall@K、候选内理想重排上界 → 检查 qrels/guidance/documents 字段。

**结果**

| 领域/split | 官方 BM25 | 发布值 | 预建索引 BM25 | BGE dense |
|---|---|---|---|---|
| IOTA dev (3) | 0.3309 | 0.3309 | 0.3852 | 0.4712 |
| IOTA train (7) | 0.1563 | 0.1563 | 0.0352 | 0.2131 |
| Bitcoin dev (30) | 0.0370 | 0.0370 | 0.0300 | 0.2171 |
| Bitcoin train (70) | 0.0856 | 0.0856 | 0.0468 | 0.1550 |

**解释**：官方基线与预建索引的 analyzer 完全相同（`io.anserini.analysis.DefaultEnglishAnalyzer`，分词逐 token 一致），差异来自 BM25 打分实现；因此预建索引不能声称"复现基线"。
**教训**：pytrec_eval 在本机不可编译，改用官方 `ir_metrics.py`，并用四个基线逐位复现证明等价。

## 4. Phase 1：cross-encoder 精排（含一次结论修正）

**初始假设**：dense 召回 0.5688、候选上界 0.6488，所以"强语义精排"应能吃到这 0.43 的空间。

**实验 A（错误版）**：`run_rerank.py`，bge-reranker-base，query+doc 直接喂 512 token，`longest_first` 截断。
结果：0.1195（dense 0.2171）。当时结论写成"通用 CE 不迁移"。

**问题**：a) RETECO query 很长（中位 221 token，最长 1017），doc 实际预算只有 ~250 token；b) 结论只基于一个 checkpoint、一个深度。

**实验 B（修正截断）**：`run_chunk_rerank.py`，显式 q_cap + 文档窗口 + max-pool。

| 配置 (Bitcoin dev) | nDCG@10 |
|---|---|
| bge-base 原始 | 0.1195 |
| bge-base 单窗口 380 | 0.1089 |
| bge-base chunk 380/190 | 0.1067 |
| bge-base 标题式 query | 0.1470 |
| MiniLM-L6 全 query | 0.1704 |
| MiniLM-L6 标题式 | 0.2041 |
| v2-m3 全 query | 0.1577 |
| **v2-m3 标题式** | **0.2121**（Δ −0.005，CI [−0.081,+0.071]） |
| **RRF(dense, v2-m3) 2:1** | **0.2311** |

**结论修正**：不是"CE 不迁移"，而是"强 CE 追平 dense，弱 CE 显著变差"；损失是全量替换造成的深度假象（depth 5→0.2143，100→0.1195）；截断和长度偏差都被数据否定（gold 96.4% token 可见，within-query 长度相关≈0）。

**关键洞察（来自代理审计）**：within-query AUC：dense 0.760 vs bge-reranker 0.668——重排器对 gold 的判别力低于它要重排的检索器。

**错误记录（必须保留）**：
1. MiniLM 第一次跑出 0.0043，原因是 `vocab.txt` 未下完（词表 5 个 token，全 `[UNK]`）；修好后 0.1704。→ 下载后必须校验 tokenizer/模型文件完整性。
2. 早期 union 文件里 recall 用 top-100、oracle 用 top-1000，报告列未标深度；已全部改为带深度字段并加断言。
3. `diag_length.py` 原用"独立 token >510"判断截断，低估了 pair 共享窗口的真实截断率（14% vs 24.7%），已在报告中采用后者。

## 5. Phase 2：跨领域验证 + listwise/pairwise 精排（已完成）

**设计**

* 第三领域：**Economics**（74,692 文档，83 条 query，非加密货币，规模与 Bitcoin 接近）。
* 冻结流程：同一 bge-base 编码器 / 同一 qrels 评分 / 同一 bootstrap；BM25 用 Anserini 本地建索引。
* listwise：RankGPT 式滑窗（window=20, stride=10），候选 = dense top-20；本地 Qwen2.5-1.5B-Instruct，零样本、temperature 0；输出必须是 1..n 的排列，解析失败按原序回退并计数。
* 对照：dense、pointwise v2-m3（标题式）、listwise、RRF。

**结果**

| 系统 | Economics dev (25) | Economics train (58) |
|---|---|---|
| BM25 (Lucene) | 0.0269 | 0.0361 |
| BGE dense | **0.1891** | **0.1209** |
| pointwise v2-m3（标题式） | 0.1664 | — |
| oracle@100 | 0.6460 | 0.6008 |

LLM 精排（Qwen2.5-1.5B-Instruct）：

* **listwise（window 20）在 IOTA/Bitcoin 上输出 = 输入顺序**（0.4712 / 0.2171，与 dense 完全相同）；原始输出就是 `[1] > [2] > ...`，属于小模型的 primacy 复读，是方法失败而非结论；
* **pairwise（top-10 全比较）**：Bitcoin dev 0.2171 → **0.2602**（+0.0431，CI [−0.021,+0.116]）；Economics dev 0.1891 → **0.2082**（+0.0191，CI [−0.012,+0.055]）；合并 55 条 **+0.0322（CI [−0.007,+0.077]，P=0.945）**；与 dense 顺序的位置一致率仅 12.7%，确有重排；
* 成本：45 次 LLM 调用/query，Bitcoin dev 353 s/1350 次 = **11.8 s/query**（Economics 11.2 s/query），约为 dense 检索（4.5 ms）的 2600 倍；
* listwise 的"复读"已用三项证据确认：原始输出就是 `[1] > [2] > ...`；解析失败 **0/30**；**把候选顺序整体反转后仍输出 `[1]..[20]`**（即输出的是输入位置而非相关性）；
* 结论：小模型有 pairwise 判别力（探针 71.7% 准确率、位置偏差 51.7%），但在该 checkpoint+prompt 下不能做 listwise；要验证 listwise 需要更大模型或 API，成本仍是问题。
* 注意：pairwise 是 depth-10 实验，pointwise 是 depth-100 实验，两者不可当作同一比较；与 dense 的比较各自都在相同候选集上。

## 5b. Phase 3：基线复现根因 + 时间特征消融（已完成）

**A. 为什么预建 Lucene 索引复现不了发布基线**

结论：官方基线用的是 gensim `LuceneBM25Model`，它把模型也应用到 query 上一次，
于是点积里 **idf 出现两次**（外加 query 长度归一化）；Lucene 只应用一次。
两者 analyzer 相同、idf 公式恒等、k1/b 相同，所以差异完全来自打分函数。

证据：
* 官方流水线复现发布值逐位一致（0.0370/0.3309/0.1563/0.0856）；
* 自建 Anserini 索引（同 jar、同参数）与官方预建索引分数**完全相同**（Bitcoin dev 均为 0.0300）→ 索引不是问题；
* 同一 query/文档：官方 93.5363，gensim 公式手算 93.4127；Lucene 77.2161，用索引自身统计手算 77.1619，Anserini `compute_query_document_score` 77.2161 —— 两个公式各自被复算验证；
* 查询内词权重跨度 13.5 倍，idf 平方显著改变词项相对权重 → top-10 重合仅约 60%。

细节见 `results/BASELINE_FORENSICS.md`。

**B. 时间特征消融**

设计：冻结 dense top-100；特征 = 年份/时间锚点/guidance 标记；对照 = 长度与词面重叠；
logistic regression 只在 train 拟合，dev 只报告一次。

| 配置 | Bitcoin dev | Δ | Economics dev | Δ |
|---|---|---|---|---|
| dense | 0.2171 | — | 0.1891 | — |
| dense_only（自检） | 0.2171 | 0 | 0.1891 | 0 |
| dense + time | 0.1941 | −0.023 | 0.2123 | +0.023 |
| dense + controls | 0.1817 | −0.035 | 0.1621 | −0.027 |
| dense + time + controls | 0.1737 | −0.043 | 0.1718 | −0.017 |

结论：方向在域间变号、CI 都跨 0；对照特征两域都变差。**内容时间（年份/锚点）特征不可靠**，
若继续做时间模块必须换成语义层时间理解。细节见 `results/TIME_ABLATION.md`。

**C. 环境事故记录（重要）**

pyserini 在 `import pyserini.search` 时会从 raw.githubusercontent.com 拉取
prebuilt-index 元数据；本机该域名不可达（或间歇超时），导致 `LuceneSearcher` 偶发
`URLError: handshake timed out`。已在 site-packages 的
`pyserini/prebuilt_index_info.py` 给两处 `urlopen` 加了 offline fallback（失败返回空表），
因为本项目从不使用 prebuilt 名称，只读本地索引。改动只影响元数据查询，不影响检索。

## 5c. Phase 4：语义时间一致性可行性检验（已完成，负结果）

**问题**：没有发布时间元数据时，"文档是否讨论 query 要求的时间/事件"这种语义时间信号，
能否在 dense 之上提供增量？

**设计**：用 guidance 的 `query_temporal_events`/`key_time_anchors`/`query_summary`
构造时间陈述 → bge-base 余弦打分；全部 dense top-100 候选、不做负采样；
query 分组的 5 折 CV + 逐 query AUC + cluster bootstrap；**placebo 对照**（错排其他 query 的陈述）；
事前门槛 ΔAUC ≥ 0.05–0.08。

**结果**

| 指标 | Bitcoin dev | Economics dev |
|---|---|---|
| 单分数 AUC：dense | 0.753 | 0.769 |
| 单分数 AUC：temporal_cos | 0.612 | 0.648 |
| placebo temporal_cos | 0.389 | — |
| 逐 query AUC（temporal 均值） | 0.693（CI [0.593,0.782]） | 0.625（CI [0.531,0.717]） |
| CV AUC dense → +temporal | 0.7388 → 0.7389（Δ +0.00005） | 0.7535 → 0.7600（Δ +0.0065） |
| 逐 query 合并 ΔAUC（胜率） | −0.0062（29.6%） | +0.0008（47.6%） |

**判断**：语义时间陈述确有 query 特定信号（真 0.69 vs 假 0.38），但与 dense 高度重叠，
增量远低于门槛 → **按事前约定不做时间重排**。这与文献一致：无时间戳文本场景缺少可比基线，
已有强方法（TempRALM 等）都依赖 passage 时间戳。

**Agent 协作**：批判代理指出"指导语泄漏/主题混淆、负样本压缩方差、AUC 不足、功效不足"，
据此加了 placebo、全量候选、逐 query AUC 与 cluster bootstrap，并事前写死效应量门槛；
调研代理确认文献现状（详见 `results/TEMPORAL_SEMANTIC_PROBE.md`）。

## 5d. Phase 5：query 表示对一阶段检索的影响（已完成）

**问题**：Phase 1 里标题式 query 让 CE 变好，那检索器是否也该用短 query？

**设计**：文档向量复用缓存（完全不动），只重算 query 向量；四种视图
full / title / guidance summary / title+summary；6 组（Bitcoin、Economics、IOTA × dev、train）；
配对 bootstrap。

**结果（nDCG@10，full 为基线）**

| 领域/split | full | title | summary | title+summary |
|---|---|---|---|---|
| Bitcoin dev | **0.2152** | 0.1543 | 0.1230 | 0.1447 |
| Bitcoin train | **0.1647** | 0.1477 | 0.1545 | 0.1266 |
| Economics dev | **0.1768** | 0.1059 | 0.1005 | 0.1057 |
| Economics train | 0.1262 | 0.1064 | **0.1495** | 0.1199 |
| IOTA dev | **0.4553** | 0.3648 | 0.2072 | 0.2937 |
| IOTA train | **0.2083** | 0.1749 | 0.0823 | 0.1606 |

**结论**：一阶段检索**必须用完整 query**；短 query 的收益只存在于重排器
（因为它对长 query OOD）。唯一例外 Economics train 的 summary（+0.023，CI 跨 0，
同域 dev 反向），不作为稳定收益。实践上：检索用 full，精排用 title/摘要视图。

**附**：单纯去 HTML 标签对检索无稳定收益（±0.01，方向不一），不能当作改进。

## 5e. Phase 6：三项工程化（已完成）

**① 强检索器 bge-large-en-v1.5（结论：不采用）**

| split | bge-base | bge-large | Δ | 95% CI |
|---|---|---|---|---|
| Bitcoin dev | 0.2171 | 0.1569 | **−0.0602** | [−0.107,−0.022] |
| Bitcoin train | 0.1550 | 0.1673 | +0.0123 | [−0.014,+0.040] |
| Economics dev | 0.1891 | 0.1835 | −0.0056 | [−0.068,+0.058] |
| Economics train | 0.1209 | 0.0918 | −0.0290 | [−0.062,+0.001] |
| IOTA dev (3) | 0.4712 | 0.5856 | +0.1145 | 样本 3 条 |
| IOTA train (7) | 0.2131 | 0.2381 | +0.0250 | 样本 7 条 |

四个大样本格子三个变差（Bitcoin dev 显著），Bitcoin+Economics 合并加权 Δ = −0.015。
排除了 fp16 数值问题（fp32/fp16 向量余弦 = 1.00000）、instruction/pooling/max_length 差异、
表示崩溃（R@100 0.569→0.512）。结论：MTEB 更强不等于本任务更强，不升级检索器。
编码成本：约 59 doc/s，三个语料共 ~45 分钟。

**② 低成本 pairwise（top-5，10 次 LLM 调用/query）**

| 领域/split | dense | top-5 重排 | Δ | 95% CI | 好/差/平 | 延迟 |
|---|---|---|---|---|---|---|
| Bitcoin dev (30) | 0.2171 | **0.2670** | +0.0499 | [−0.021, +0.129] | 6/5/19 | 2.7 s |
| Bitcoin train (70) | 0.1550 | 0.1437 | −0.0113 | [−0.032, +0.009] | 8/11/51 | 2.4 s |
| Economics dev (25) | 0.1891 | **0.2026** | +0.0135 | [−0.004, +0.036] | 4/3/18 | 2.7 s |
| Economics train (58) | 0.1209 | 0.1273 | +0.0065 | [−0.009, +0.026] | 5/6/47 | 2.6 s |

成本从 11.8 s 降到 ~2.5 s（4.6×），但 **Bitcoin dev +0.050 与 train −0.011 方向相反**，
收益不稳定。参考天花板（只重排 top-5 时最多可提升）：Bitcoin dev +0.1029、train +0.0521、
Economics dev +0.0452、train +0.0331；实际捕获 48% / 0% / 30% / 20%。

**③ 两视图流水线（检索 full + 精排 title + RRF 2:1）**

| 领域/split | dense | 两视图 | Δ | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|
| Bitcoin dev | 0.2171 | **0.2311** | +0.0140 | [−0.030, +0.052] | 0.75 |
| Bitcoin train | 0.1550 | **0.1713** | +0.0163 | [−0.009, +0.043] | 0.90 |
| Economics dev | 0.1891 | 0.1921 | +0.0030 | [−0.024, +0.032] | 0.58 |
| pooled (125) | — | — | **+0.0131** | [−0.005, +0.032] | 0.92 |

三格方向一致为正、成本仅一次 CE 前向（~0.7 s/query），是目前最稳的可用组合，但仍未显著。

## 5f. Phase 7：跨 13 领域两视图流水线 + 时间表示（已完成）

**问题**：Phase 6 的两视图融合只在 Bitcoin/Economics 上验证过（+0.013，未显著）。
它是否普遍？固定权重 2:1 是否合理？时间约束能不能单独表示出来？

**跨领域两视图**（12 领域 516 query，dev）：

| 方法 | Δ vs dense（query 级） | 95% CI | 域级宏平均 Δ | 域级整群 CI |
|---|---|---|---|---|
| CE 单独 | +0.0344 | [+0.0143,+0.0545] | +0.0103 | [−0.027,+0.045] |
| RRF 2:1 | +0.0305 | [+0.0212,+0.0401] | +0.0215 | [+0.009,+0.034] |
| RRF 1:2 | +0.0423 | [+0.0265,+0.0582] | +0.0239 | [−0.008,+0.053] |
| RRF 1:4 | **+0.0461** | [+0.0288,+0.0635] | **+0.0258** | [−0.010,+0.058] |

* LOOCV：1:4 在 10/12 折被选中，平均留出增益 **+0.0150**；
* 失效领域：law −0.090、workplace −0.093（两者 CE 单独就低于 dense）；
* 结论：CE 主导融合稳定为正，但"换新领域的诚实增益"只有 +0.015，且存在反向领域。
  2:1 虽增益较小，但域级方差最小——保守配置。

**时间表示**（三种，全部未超过 dense）：

| 方法 | Bitcoin dev | Economics dev | dense 基线 |
|---|---|---|---|
| steps oracle RRF（最优 w0） | 0.2393 (w0=4) | 0.1652 (w0=8) | 0.2171 / 0.1891 |
| LLM 分解（Qwen2.5-1.5B） | 0.2012 | 0.1613 | 同上 |
| HyDE（α=1.0） | 0.1819 | 0.1529 | 同上 |

代价：LLM 分解 35 次调用/query；HyDE 需生成 5 段假设文档。均无增益 → 不做。

**数据事故**：workplace / politics 的 `examples_dev.jsonl` 被 hf-mirror 的 Cloudflare
错误页覆盖（7,030 字节 HTML）、`guidance_dev.jsonl` 为 0 字节，导致 dense 阶段
JSONDecodeError。已按 DOCTYPE 校验重下并补跑（`scripts/phase7_catchup.sh`）。

**产物**：`results/PHASE7_CROSS_DOMAIN.md`、`results/TEMPORAL_DECOMPOSITION.md`、
`results/phase7_summary_all13.json`。

## 6. Agent 协作记录

| 代理 | 任务 | 关键发现 | 处置 |
|---|---|---|---|
| 审计代理 | 用自写实现重算 nDCG、检查候选包含 | 数字全部对得上；rerank 是 dense top-100 的精确置换 | 采纳 |
| 批判代理 | 代码与方法学审查 | 共享窗口截断、深度假象、AUC 0.668<0.760、显著性缺失、query 形式未测 | 全部做了对应实验 |
| 调研代理 | 长文档 CE 实践 | 官方就是 512 截断；chunk 收益通常 <5%；建议按长度分层归因 | 与实测一致 |
| Phase 2 代理 | listwise 方法学调研 | （第二次调用失败，未产出） | 改为自建 RankGPT 协议 |

**协作原则**：代理只做只读分析/独立实现，不写项目文件；它们的结论必须能被我复算或由 run 文件支撑；与代理结论冲突时以可复现实验为准。

## 7. 待办

1. ~~Economics 全流程与 `PHASE2_REPORT.md`~~ 已完成（§5）。
2. ~~时间特征消融~~ 已完成（§5b，负结果）。
3. ~~跨 13 领域两视图 + 时间表示~~ 已完成（§5f）。
4. 未决：CE 失效领域（law / workplace）的门控或跳过策略；测试集提交格式与 run 校验；
   系统论文初稿（任务描述、方法、跨领域表格）。

## 8. 文件索引

- `results/PHASE0_REPORT.md`：Phase 0 全表
- `results/RERANK_NOTE.md`：Phase 1 全配置与统计
- `results/AGENT_REVIEWS.md`：代理结论与处置
- `results/PHASE2_REPORT.md`：Phase 2（含 12 条审阅修正表）
- `results/BASELINE_FORENSICS.md`：官方基线复现根因
- `results/TIME_ABLATION.md`：时间特征消融
- `results/TEMPORAL_SEMANTIC_PROBE.md`：语义时间一致性（负）
- `results/QUERY_REPRESENTATION.md`：query 表示变体
- `results/PIPELINE_AND_TOP5.md`：Phase 6 三项工程化
- `results/TEMPORAL_DECOMPOSITION.md`：时间分解 + HyDE（负）
- `results/PHASE7_CROSS_DOMAIN.md`：跨 13 领域两视图（最终）
- `scripts/`：`run_bm25_pyserini.py`、`run_dense_bge.py`、`run_chunk_rerank.py`、`run_listwise.py`、`score_run.py`、`oracle_sweep.py`、`union_recall.py`、`make_report.py`、`package_results.py`、`phase7_all_domains.sh`、`phase7_catchup.sh`、`analyze_phase7.py`
