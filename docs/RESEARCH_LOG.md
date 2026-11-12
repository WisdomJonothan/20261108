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
- `results/PHASE8_STRONGER_BASELINES.md`：强基座对比 + mxbai 两视图（Phase 8）
- `results/TEMPORAL_OBSERVABILITY.md` / `TEMPORAL_CEILING.md`：时间可观测性与上限
- `scripts/`：`run_bm25_pyserini.py`、`run_dense_bge.py`、`run_chunk_rerank.py`、`run_listwise.py`、`score_run.py`、`oracle_sweep.py`、`union_recall.py`、`make_report.py`、`package_results.py`、`phase7_all_domains.sh`、`phase7_catchup.sh`、`analyze_phase7.py`

### 5g. Phase 8：强基座 + 时间可观测性（2026-11-08）

外部审阅指出"没跟强模型比"，核对后：TEMPO 的 E5 30.4 是 e5-mistral-7b（7B）、
DiVeR 32.0 是 4B 模型，且是 1,730 query / v1.0 语料；与我们的 519 dev / v1.1 不可
并排。能跑的替代：e5-base-v2（13 域）、mxbai-embed-large-v1（12 域）、gte-large
（transformers 5.22 不兼容，放弃）。

结果（dev nDCG@10 宏平均）：e5-base 0.1489、bge-base 0.2488、**mxbai 0.2995**
（12 域）。用 mxbai 作第一阶段，bge-reranker-v2-m3 标题窗口重排 + RRF 1:1 →
**0.3246**；配对 bootstrap +0.0333 [+0.0188,+0.0479]，domain-clustered +0.0252
[+0.0027,+0.0506]（首次 CI 不含 0），LOOCV +0.0160。

时间侧（`RESULTS/TEMPORAL_CEILING.md`）：把 top-100 中"年份与查询区间不一致"的
文档降权，即使是 oracle 排序，在 8 个域里有 5–6 个 nDCG 下降（politics 0.307→0.268、
genealogy 0.211→0.184、hsm 0.293→0.271、history 0.277→0.269、bitcoin 0.217→0.197），
只有 economics 上升（0.189→0.201）。原因：有年份的候选里 53–100% 年份不在查询区间，
其中很多是 gold。→ "时间不一致 ≠ 不相关"，这是此前全部时间模块负结果的根因。

文件：`results/PHASE8_STRONGER_BASELINES.md`、`results/TEMPORAL_OBSERVABILITY.md`、
`results/TEMPORAL_CEILING.md`、`scripts/phase8_baselines.sh`、
`scripts/phase8_mxbai_rerank.sh`、`scripts/analyze_mxbai.py`、
`scripts/temporal_coverage.py`、`scripts/temporal_ceiling.py`、
`scripts/hardneg_mine.py`。

## 5h. Phase 9：数据缺口修复 + V1 冻结 + 监督式交叉编码器（进行中）

**A. 数据事故：所谓"只有 8 个领域有 train"是下载不完整**

此前所有推理都基于"13 个领域里只有 8 个有 train 分片"，因此监督训练看起来只能
覆盖 60% 的评测面。核对官方 data 页后：Track-1 为 **1,211 train / 519 dev**，
按域 70/30 切分。本地缺的正是 cardano(36)、law(24)、quant(24)、travel(70)、
workplace(25) 五个域的 `examples_train.jsonl` / `guidance_train.jsonl` /
`qrels_train.txt` / `steps_*.jsonl`。用 `scripts/fetch_missing_train.py`
（列目录 → 比对大小 → 下载 → 逐行 JSON/QRELS 解析校验）补齐 58 个文件后，
本地计数为 1,211 / 519，与官方完全一致。

同时确认语料已是 **v1.1**（13 域文档总数 1,167,159，与官方去重后总数逐位相同）。

**B. V1 冻结与"两种宏平均"**

`fuse_rrf.py` 融合出的 13 个 dev run 补齐后（`runs/sub_1_1_<domain>_dev.trec`），
用官方 `format_checker.py`（带 `--qrels`/`--corpus`）逐文件校验：13/13 通过，
0 error。为解决对称 RRF 造成的平分（519 个 topic 中 78 个 top-10 内有平分），
写出时按 `score − 1e-9·rank` 去平分，排名顺序不变。

| 聚合方式 | nDCG@10 |
|---|---:|
| topic-macro（项目旧口径，519 topics） | **0.3199** |
| domain-macro（官方口径） | 0.3166 |
| topic-macro，n≥19 的 8 个域（469 q） | 0.3243 |

教训：同一个 run 集给出两个不同数字，报告必须写明口径。官方口径是
domain-macro（出处与更正见 §5i，早先这一行把 topic-macro 当作官方，是误读）。
另修复 `score_run.py`：此前对 run 中缺失的 topic 只对"存在的 topic"求均值，
官方把缺失 topic 记 0；现默认按 0 计入（`--allow-missing` 可恢复旧行为）。

**C. 融合方式：RRF vs z-norm 分数和**

RRF 丢弃分数幅度、按名次合并；当一个是明显更强的检索器时会被"融合税"拖累。
`scripts/fuse_scores.py` 在查询内对两路分数做 z 标准化后加权求和
（候选集不变，recall 不变）：

| 融合 | topic-macro | domain-macro | 配对 bootstrap（query） |
|---|---:|---:|---|
| RRF 1:1（V1） | 0.3199 | 0.3166 | — |
| score-sum 1:2 | **0.3302** | 0.3193 | +0.0103 [+0.0021,+0.0184]，域整群 CI 跨 0 |
| score-sum 1:1 | 0.3260 | 0.3249 | +0.0061 [−0.0020,+0.0145] |
| score-sum 2:1 | 0.3202 | 0.3208 | — |
| score-sum 1:4 | 0.3202 | 0.3017 | — |

关键现象：**按 topic-macro 选，CE 该给更大权重（1:2）；按 domain-macro 选，
最优是 1:1**。这正是"在 dev 上选权重"的风险，所以权重必须在 train 上拟合。
（用 7 个有 train 分片的域做域内权重：dev 0.3082 < 固定 2:1 0.3179，方向为负，
说明"按域调权重"不可行；但"全局一个 scalar 在 train 上拟合"尚未做。）

**D. 新增工具**

| 脚本 | 作用 |
|---|---|
| `scripts/fetch_missing_train.py` | 与镜像目录比对、补齐缺失分片、校验可解析性 |
| `scripts/validate_submission.py` | 官方 checker + topic 集合相等 + 语料归属 + 连续名次 + 去平分 |
| `scripts/aggregate_scores.py` | 同时给出 topic-macro / domain-macro / n≥19 子集 |
| `scripts/bootstrap_compare.py` | 逐 query 与按域整群两种配对 bootstrap |
| `scripts/fuse_scores.py` | z-norm 分数和融合（可指定权重与归一方式） |
| `scripts/build_ce_train_groups.py` | 从 train 候选构造 listwise 训练组（正例全留 + 70% 难负例） |
| `scripts/train_ce.py` | 交叉编码器微调（listwise、bf16、梯度检查点、train 留出校验、AUC） |

**E. 监督微调（进行中）**

诊断先行：冻结 top-100 候选池的 oracle nDCG@10 是 0.745，V1 只拿到 0.320——
约 5% 的排序空间；而零样本交叉编码器的 within-query AUC 是 0.801，与它要重排的
dense（0.806）几乎一样。**换后处理不会动这个量，只有改打分器本身才行**，这是
此前十个"聪明干预"全部归零的统一解释。

数据：13 域 train，1,071 个 query 组、2,876 个正例、18,544 个负例（70% 为难负例）。
首个 pilot（bge-reranker-base，1 epoch）：train 留出 0.2240 → 0.2301（+0.006），
说明训练回路有效但该模型太弱；正式实验换 bge-reranker-v2-m3（568M，bf16 + 梯度
检查点，峰值 5.5 GB），零样本 train 留出基线 0.2535 / AUC 0.785。

预注册的采纳门槛：dev topic-macro ≥ V1 + 0.015 且域整群 CI 排除 0；否则只作为
消融，提交 V1。且 SFT 必须与"在 train 上重新拟合融合权重"一起评估——模拟显示
CE 单独的提升只有约 19%（RRF 1:1）、26%（score-sum）、61%（RRF 1:4）能穿过融合。

**F. V1.1：融合配置在 train 上拟合，dev 只读一次**

为避免"在 dev 上挑权重"，把融合形式与权重放进一个网格
（rrf / z-norm 分数和 × 4:1, 2:1, 1:1, 1:2, 1:4, 1:8），**全部在 train 的
1,211 条 query 上**打分选优（`scripts/fit_fusion_weight.py`）：

| 形式 | 权重 | train topic-macro | train domain-macro |
|---|---|---:|---:|
| **z** | **1:2** | **0.2893** | 0.2596 |
| z | 1:1 | 0.2868 | 0.2612 |
| rrf | 1:2 | 0.2799 | 0.2519 |
| rrf | 1:1（旧 V1） | 0.2782 | 0.2485 |
| rrf | 2:1 | 0.2758 | 0.2482 |

train 上选出的最优 = **z-norm 分数和 1:2**，与 dev 上的最优一致（dev 上
z 1:2 = 0.3302）——说明这个选择不是 dev 拟合出来的。把它应用到 dev，只读一次：

| 系统 | dev topic-macro | dev domain-macro | 配对 bootstrap（query） | 域整群 |
|---|---:|---:|---|---|
| V1（RRF 1:1） | 0.3199 | 0.3166 | — | — |
| **V1.1（z 1:2）** | **0.3302** | 0.3193 | **+0.0103 [+0.0021,+0.0184]** | [−0.0131,+0.0175] |

n≥19 的 8 个域子集：V1 0.3243 → V1.1 **0.3365**。13 个 run 全部通过官方
format_checker（带 --qrels/--corpus），并做去平分后冻结为
`runs/sub_1_2_<domain>_dev.trec`。结论：V1.1 取代 V1 作为提交配置；但按域整群
口径（13 个域重采样）仍未显著，这个限制必须与数字一起报告。

**G. 监督微调的第一批 dev 结果（1 epoch，13 域）**

训练侧门控（留出 45 条 train query）：AUC 0.7950 → 0.7990（+0.0041，未达 +0.01 门槛），
nDCG@10 0.2870 → 0.3008（+0.0138）。按预注册规则"门控不过 → 只作消融"，但为论文
仍需 dev 数字，故继续做了一次 dev 重排（不是采纳尝试）。

| 系统（dev，519 topics） | topic-macro | domain-macro | Δ vs 零样本 |
|---|---:|---:|---:|
| 零样本 CE 单独 | 0.2990 | 0.2680 | — |
| **SFT CE 单独** | **0.3096** | 0.2742 | **+0.0106** |
| V1.1 = dense + 零样本 CE，z 1:2 | 0.3302 | 0.3193 | — |
| **SFT = dense + SFT CE，z 1:2** | **0.3418** | 0.3274 | **+0.0116** |

SFT 融合 vs V1.1 的配对 bootstrap：query [+0.0069,+0.0165]；**域整群 [+0.0040,+0.0123]
—— 这是本项目第一个域整群 CI 排除 0 的改进**。逐 topic：125 好 / 58 差 / 336 平；
逐域 11/13 提升（只有 bitcoin −0.003、monero −0.001 轻微下降）。

注意两点：(1) 融合权重 1:2 是从**零样本** CE 上拟合后继承的，没有为 SFT CE 重拟合
（重拟合需要在 train 上再跑 1,211 条 CE，约 2 小时，留待后续）；(2) 预注册的采纳门槛是
≥ +0.015，实测 +0.0116，**差 0.0034 未达标**，因此严格按协议不能替换提交件；
但域整群 CI 首次排除 0 是更强的证据，最终取舍单独记录。

产物：`runs/sub_sft12_<domain>_dev.trec`（13/13 官方格式校验通过，去平分），
`results/aggregate_sub_sft12_dev.json`、`results/boot_sft_vs_v11.json`。

## 5i. Phase 9d：官方口径更正（domain-macro）+ SFT 正式采纳 + 工程审计修复

**触发**：外部审阅指出项目一直在用 topic-macro 报成绩，而官方聚合是 domain-macro。
核对 Evaluation Plan（https://datascienceuibk.github.io/RETECO/evaluation.html）原文：
"nDCG@10 is computed for each query, averaged over the queries of each domain,
and then averaged over the 13 domains, so every domain counts equally"，
且 "missing queries ... receive zero"。**结论：官方指标 = domain-macro，早先按
topic-macro 选融合权重是错的。**

### 影响一：融合权重必须按 domain-macro 选

`fit_fusion_weight.py` 增加 `--select-metric`（默认 `domain_macro`）并在 train
（1,211 条）上重拟合：**z 1:1 胜出**（train domain-macro 0.2612 vs 1:2 的 0.2596）。
因此 1:2 的 "V1.1" 作废，官方口径下的零样本基线是 **z 1:1 = 0.3249**。

### 影响二：SFT 增益在正确口径下更大（但登记的 topic-macro 门槛未达，见下）

| 系统（dev, 519 topics） | domain-macro（官方） | topic-macro |
|---|---|---|
| 官方 BM25 | 0.1147 | — |
| dense mxbai（top-100） | 0.2943 | 0.2891 |
| 零样本 CE 单用 | 0.2680 | 0.2990 |
| SFT CE 单用 | 0.2742 | 0.3096 |
| V1 = dense+zs CE, RRF 1:1 | 0.3166 | 0.3199 |
| dense+zs CE, z 1:1 | 0.3249 | 0.3260 |
| dense+zs CE, z 1:2（作废） | 0.3193 | 0.3302 |
| dense+SFT CE, z 1:2 | 0.3274 | 0.3418 |
| **dense+SFT CE, z 1:1（采用）** | **0.3336** | 0.3339 |

配对 bootstrap（10k 次，seed 7，统计量=官方 domain-macro）：

- SFT vs 零样本（固定 z 1:1，只换 CE）：+0.0087，按查询重采样 CI [+0.0025,+0.0151]，
  按域整群重采样 CI [+0.0037,+0.0142]（两者都排除 0；95 好 / 55 坏 / 369 平）；
- z 1:1 零样本 vs V1 RRF（只换融合形式）：+0.0083，CI 跨 0（弱证据）；
- **SFT z 1:1 vs V1 RRF（总效果）：+0.0170**，查询 CI [+0.0055,+0.0290]，
  域整群 CI [+0.0052,+0.0287]；169 好 / 106 坏 / 244 平。

**关于"预注册门槛"的更正（不要写成"已通过"）**：门槛登记时的统计量是
topic-macro（dev tag ≥ V1 + 0.015 且域整群 CI 排除 0）。按登记时的统计量，
SFT − V1 = 0.3339 − 0.3199 = **+0.0141，未达到 +0.015**；若对 1:2 权重则是
+0.0116。这个门槛是在口径更正（§5i）之后才"看起来通过"的，论文里必须如实写
"登记的 topic-macro 门槛未达"，不能写"通过预注册门槛"。

可以站得住的表述是：按官方 domain-macro，**只替换 CE（相对最强零样本基线
z 1:1，0.3249）的增益 +0.0087，域整群 CI [+0.0037,+0.0142] 是唯一排除 0 的
单因素效应**；+0.0170 是它与融合形式变化（+0.0083，CI 跨 0）的合计，不能整体
记在监督训练名下。之前"差 0.0034 未达标"的判断则是 topic-macro + 1:2 权重的
双重假象，同样需要更正。冻结提交件
**`runs/sub_sft11_<domain>_dev.trec`**（13/13 官方格式校验 0 错误，去平分后
二次校验）。

### 工程审计修复（对应外部审阅 P0）

1. `score_run.py` 已改：缺失 query 记 0（原先静默丢弃），除非显式 `--allow-missing`；
   已同步进仓库（此前 GitHub 上是旧版）。
2. `validate_submission.py --detie` 现在**对去平分后的文件重新校验**（topic 集合、
   严格递减、有限值）。同时修掉去平分公式的缺陷：旧式 `score - 1e-9*rank` 在
   `-1e9` 填充哨兵上被浮点吞掉，导致 100 名之后仍并列；改为按数量级缩放的
   序数去平分（只处理并列，保留原顺序）。
3. `test_mode_dryrun.sh` 增加存在性保护：`examples_test.jsonl` 已存在则拒绝，
   需 `FORCE=1` 才覆盖，避免删掉真实测试集。
4. 新增 `requirements.txt`（锁定训练解释器版本）、`results/MODEL_MANIFEST.json`
   （三个模型的 HF repo、参数量、`model.safetensors` sha256）、
   `scripts/reproduce_all.sh`（一键：dense→CE→z 1:1→校验→打分→聚合）。
5. 语料指纹 `results/corpus_manifest.json` 已在库中（13 域 / 65 文件 / 1,167,159 文档）。

### §5j 权重敏感性：+0.0087 依赖融合权重吗？（2026-11-09 下午）

审稿人真正会问的不是"最优权重是多少"，而是"+0.0087 是不是权重挑出来的"。为此写了
`scripts/weight_sweep.py`（纯 CPU，33 秒）：在 dev 上把**两条臂**都按
`w ∈ {0.5,0.7,0.85,1,1.15,1.3,1.5,1.8,2}` 重新融合，报告零样本 / SFT / 差值
及**域聚类 bootstrap** 95% CI。仅作诊断，不在 dev 上选任何参数。

| w(CE) | 零样本 | SFT | Δ | 域聚类 95% CI | 上行域 |
|---:|---:|---:|---:|---|---:|
| 0.5 | 0.3208 | 0.3211 | +0.0004 | [−0.0031,+0.0042] | 5/13 |
| 0.7 | 0.3294 | 0.3287 | −0.0007 | [−0.0081,+0.0053] | 7/13 |
| 0.85 | 0.3261 | 0.3308 | +0.0047 | [+0.0001,+0.0098] | 8/13 |
| **1（冻结）** | 0.3249 | **0.3336** | **+0.0087** | **[+0.0038,+0.0142]** | 10/13 |
| 1.15 | 0.3271 | 0.3319 | +0.0048 | [+0.0008,+0.0089] | 9/13 |
| 1.3 | 0.3278 | 0.3334 | +0.0056 | [+0.0004,+0.0114] | 10/13 |
| 1.5 | 0.3253 | 0.3298 | +0.0045 | [−0.0018,+0.0108] | 8/13 |
| 1.8 | 0.3192 | 0.3341 | +0.0149 | [+0.0055,+0.0293] | 12/13 |
| 2.0 | 0.3193 | 0.3274 | +0.0080 | [+0.0041,+0.0122] | 11/13 |

结论必须如实写：**融合面在 ±0.008 量级上是平的，没有单调趋势**——零样本臂自身随 w
的波动与 SFT 带来的增益同量级，说明这些差异多为重排噪声，而非"权重敏感性"。
SFT 臂在 9 个权重里 7 个领先、在 w≥0.85 的每个权重都领先。因此论文只主张
**冻结权重（预注册、在 train 上用零样本拟合、从未在 dev 上调）下的 +0.0087**，
不主张"权重鲁棒"。同时说明：该权重的调参对象是**对照臂**，SFT 臂反而是欠调的一方，
所以这个比较只会低估、不会夸大 SFT 的收益。

另外两个必须预备的问题：
1. **"CE 单独（0.2742）低于 dense（0.2943），为什么要 rerank？"**——因为两者错误互补：
   dense 与 SFT CE 在 top-100 内的平均 Spearman 仅 **0.328**（`weight_sweep.py`
   同批算出），融合才有增益；单模型 nDCG 低不代表融合无用。
2. **"+0.0087 是否只靠一两个域"**——w=1 时 10/13 域上行，per-domain 表已落
   `results/WEIGHT_SWEEP.md`。最大正贡献是 cardano（+0.0336，36 条 query），
   负贡献是 bitcoin（−0.0080）。

### §5k 事故记录：OOF 重训用错了底座模型（2026-11-09 16:00）

做 P1 #1（用 `--dump-val-qids` 重训 SFT，以便**可证明**地把融合权重在训练集内
out-of-fold 拟合）时，`scripts/phase11_sft_oof.sh` 调用 `train_ce.py` **没有传
`--model`**，而 `train_ce.py` 的默认值是 `models/bge-reranker-base`（278 M），
不是前两个 checkpoint 用的 `models/bge-reranker-v2-m3`（567.8 M）。结果：

| 产物 | 实际模型 | 后果 |
|---|---|---|
| `results/ce_ft/v2m3_pool8_e1b/` | bge-reranker-base 278 M | 训练 29 min、日志 `params_m: 278.0` |
| `runs/rerank_sft_e1b_*_dev.trec` | 同上 | CE 单独 dev 域宏 **0.1957**，远低于零样本 CE 0.2680 |
| `runs/rerank_e1b_*_oof.trec` | 同上 | OOF 拟合权重 RRF 1:1 域宏 0.3182（train 45 query） |
| dev 套用该权重 | 同上 | **0.2799**，低于 dense 0.2943 与零样本 RRF 1:1 0.3166 |

即：这组数字衡量的是"一个更弱的底座模型"，**不能作为 SFT 有效性证据**。
所有相关文件已移动到 `results/_wrong_base_20261109/`（未删除，保留审计），
并在 `PHASE11` 报告里标注为作废。

修复（已生效）：
1. `phase11_sft_oof.sh` 的训练行显式加 `--model models/bge-reranker-v2-m3`；
2. 训练启动后必须核对日志 `data_loaded.model` 含 `v2-m3`、`params_m ≈ 567.8`，
   否则立即 kill——这条检查本身要写进脚本（见 `check_ce_model.py`）；
3. 这次重跑（16:16 启动，bash-1893）已确认 `params_m 567.8`，OOF 权重与 dev 数字
   出来后覆盖本节；若 OOF 权重下 SFT 仍不领先，则论文撤回 SFT 主张，只保留
   "零样本 CE 融合 +0.0170（域宏 0.3166 vs 0.2943）"这一条更强的结论。

教训（写进实验规则 9）：**凡是"tag 里带模型名"的实验，都要在日志里断言
`model` 与 `params_m`**；文件名/目录名不是证据，配置与日志才是。

## §5l. Phase 12：2-epoch SFT 臂（dev 域宏 0.3581）与深度探针（2026-11-09 傍晚）

### 结果

| 系统（dev 519 query，域宏） | 值 |
|---|---|
| zero-shot CE，z 1:1 | 0.3249 |
| SFT CE（1 epoch，lr 1e-5），z 1:1 | 0.3336 |
| **SFT CE（2 epochs，lr 1.5e-5，best@ep1），z 1:1** | **0.3581** |
| 2-epoch 臂 vs zero-shot（域宏，10k bootstrap） | **+0.0332**，query CI [+0.0161,+0.0523]，domain CI [+0.0192,+0.0472]，188 升 / 78 降 / 253 平 |

- 12 域（去 iota）0.3181 vs 0.2832（+0.0349）；n>=19 的 8 域 0.3267 vs 0.2985（+0.0282）→ 增益不是 iota 或小域带来的。
- CE 单独：1 epoch 0.2742 → 2 epochs 0.2823，仍低于 dense 0.2943；融合后 0.3581，说明收益来自与 dense 的互补（Spearman 0.279）。
- 权重扫描（`results/WEIGHT_SWEEP_E2.md`）：w=0.5…2.0 九个权重**全部为正**（+0.0089 … +0.0346），w=0.7/1.5/1.8 时 13/13 域全升。
- 冻结产物：`runs/sub_sft_e2bc_*`（13 域，format_checker 0 错），`results/aggregate_sub_sft_e2bc_dev.json`，`results/boot_dom_e2_vs_zs.json`。

### 2-epoch 相对 1-epoch 改了什么

两者只差 **epoch 数与峰值学习率**（1e-5 → 1.5e-5）。2-epoch 的 `best` 出现在 epoch 1，epoch 2 的 val nDCG 反而降（0.3029 → 0.3012），所以实际用的是"一个 epoch、峰值 1.5e-5、cosine 半程"的权重。因此这更像**学习率/有效训练量**效应，而不是"多训一轮"。

### 深度探针：k=300 不划算（已 kill）

- 稠密池 Recall@100 0.6995 / @200 0.7783 / @500 0.8697；oracle@10 0.7446 / 0.8150 / 0.8939。
- 取 62 个"gold 在 100–500 名之间"的难 query，用 200 深池重排（e1 checkpoint）：域宏 **+0.0010**，query CI [−0.0009,+0.0036]，9/13 域恰好 0.0000。
- 结论：CE 对 100 名以后的文档打分接近地板，加深池子只增加成本；k=300 全量重排取消（记录 `results/DEPTH_ANALYSIS.md`）。

### 工程修复与事故

1. `run_chunk_rerank.py --examples` 只替换了 query 文本、仍遍历候选 run 的全部 topic（`KeyError: '126111_0'`）；已改为按 `--examples` 过滤候选。
2. `train_ce.py` 的参数是 `--tag/--outdir`（且 `run_dir = outdir/tag`），默认底座是 `bge-reranker-base`。`train_sft_variant.sh`、`phase12_lr_sweep.sh`、`TRAINING_CONFIGS.md` 已补 `--model models/bge-reranker-v2-m3` 与正确 outdir。
3. 后台训练两次在 ~10 分钟后被宿主静默 kill（无 traceback、GPU 显存不涨）。改为 `Start-Process` 完全脱离 job 进程树后稳定运行；**以后长训练一律 detached**。

### 待办

- 种子复现：seed 23/29（detached 运行中），出来后填 `results/SEED_REPLICATION.md`。
- 论文 `docs/PAPER_DRAFT.md` 已改以 2-epoch 臂为主角（摘要/表 2/表 3/§5.1/§5.2/§4 文字）。

## 5m. Phase 14 — 种子复核（seed 29，2026-11-09）

同日完成第二个训练种子的复核，协议跑之前就写进 `results/NEXT_STEPS.md`：
平均 Δ ≥ +0.005、至少 2/3 个种子为正、域聚类 CI 排除 0。

- 训练 `v2m3_pool8_e2s29`：与冻结算法规格相同（2 epoch、peak lr 1.5e-5、freeze 8/24、
  max-pool 训练、最多 3 窗、max_len 320），只把 `--seed` 改成 29。验证集
  0.2846 → **0.3195**（epoch 1，best）→ 0.3054（epoch 2），选 checkpoint 规则与原始 run 一致。
- dev（519 查询）：域宏 **0.3527**（seed 17 为 0.3581），对 zero-shot 融合基线
  **+0.0278**，查询 CI [+0.0118,+0.0455]、域聚类 CI [+0.0159,+0.0404]，181/82/256；
  12 域（去 iota）0.3123（则 Δ=+0.0291），n≥19 0.3232（Δ=+0.0247）。
- 结论：**可复现**。两个种子的域聚类 CI 都排除 0，种子间差距 0.0054 远小于相对基线的
  增益；iota（3 条查询）不承载效应。诚实主结论从 "+0.0332" 改为
  **"+0.031 ± 0.003 over zero-shot"**，冻结系统仍报 seed 17 的 0.3581。
- 产物：`results/REPLICATION.md`、`results/aggregate_sub_sft_e2s29_dev.json`、
  `results/boot_dom_e2s29_vs_zs.json`、`runs/rerank_sft_e2s29_*_dev.trec`。
- 工程教训（已在 NEXT_STEPS §6 固化）：宿主约 10 分钟会静默杀掉后台作业；8 GB GPU 上
  任何加载模型的任务都不能与训练并行（本轮两次训练就是被并行的 watcher/复核任务打死的）。

## 5n. Phase 15 — 种子收口、Phase 11 更正与提交回放（2026-11-10）

### 种子复核完成

- seed 23（`v2m3_s23_e2`，00:58 训完，epoch 1 best：0.2754 > 0.2684）：dev 域宏 **0.3575**，
  对 zero-shot z 1:1 的 **+0.0326**，查询 CI [+0.0149,+0.0521]、域聚类 CI [+0.0184,+0.0470]，186/83/250。
- seed 29 重训（`v2m3_s29_e2`，03:49 训完，epoch 1 best：0.2838 > 0.2775）：dev 域宏 **0.3528**，
  **+0.0279**，查询 CI [+0.0119,+0.0455]、域聚类 CI [+0.0160,+0.0405]，182/83/254。
  同一 seed 的两次独立训练得到 0.3527 / 0.3528（差 5e-5），训练管线本身可复现。
- 三种子：0.3581 / 0.3575 / 0.3528，均值 **0.3561**、样本 sd **0.0029** → 主结论维持 **+0.031 ± 0.003**。

### Phase 11 的更正：过期聚合，不是 epoch 选择踩空

- 现有 `results/aggregate_sub_sft_e2b_dev.json`（tag `sub_sft_e2b`）给出 z 1:1 = **0.3583**、
  topic 0.3680；用当前 `runs/fused_sft_e2b_bitcoin_dev.trec` 重跑 `score_run.py` 得 0.214096，
  与新存档逐位相同。
- 旧的 0.3196 与当前 run 文件不一致：03:53–03:58 一轮全量重算把 score 文件按当前 run 重写过，
  旧聚合引用的仍是重算前的 score。RRF 1:1（0.3431）与 OOF RRF 1:4（0.3187）在重算后不变。
- 因此 11-09 夜间"0.3581 不可复现、z 1:1 融合形式脆弱"的结论**撤回**：
  五条独立训练（0.3581 / 0.3583 / 0.3575 / 0.3527 / 0.3528）落在 0.003 宽的区间内。
- 仍然成立的窄结论：OOF 在 45 条 train query 上选出 RRF 1:4 时增益只有 +0.0205 [+0.0010,+0.0404]，
  所以论文把固定 z 1:1 当系统、RRF 1:4 当权重受控变体。
- 审计规则（新增，写进 `results/PHASE11_VALIDATION.md` 与提交脚本）：**任何进报告的数字，
  都必须能从当前 run 文件重算出来**；聚合产物与 run 文件不同步时以 run 文件为准。

### test replay 与提交加固

- `results/test_replay/submission_{iota,law,quant}.trec` 全部通过官方 `format_checker.py`：
  iota 3000 行 / 3 topics、law 11000 / 11、quant 10000 / 10，`errors 0  warnings 0  RESULT: VALID`；
  另做严格扫描：0 处排名逆序、0 对打印同分、每 topic 正好 1000 行。
- 去平局步长原来用 `max(1e-9, abs(prev)*1e-9)`，小于 `%.6f` 打印精度，导致相邻两行打印同分；
  已改为 `max(2e-6, abs(prev)*1e-6)`。
- 两条缓存指纹链：`run_dense_generic.py` 的 doc embedding 缓存、`make_submission.py` 的 dense/CE run 缓存，
  都在旁写 `<file>.fp.json`（含模型、pooling、max_length、dtype、输入指纹）；命中比对、不符即拒绝复用，
  `--refresh` 强制重算，`--strict-cache` 连无 sidecar 的旧缓存也拒绝。每次提交另写 `<out>.config.json`。

### 工程教训

- 本会话与另一个 DSH 会话曾同时操作 D:\RETECO（同一仓库、同一批输出文件、同一块 8 GB GPU）。
  发现后杀掉了自己重复的那条队列；`aggregate_sub_*.json` 与 `runs/` 不一致也来自这次并发。
  以后同一时间只允许一个会话持有 GPU 队列与写权限。
- WDDM 下 `nvidia-smi` 常驻约 3 GB 且 compute-apps 全是 `[N/A]`，显存阈值不能用来判断空闲；
  改用 `ops/gpu_busy.ps1` 按进程命令行匹配（run_chunk_rerank / train_ce / run_dense_generic / encode_docs / score_ce）。
- `CUDA_VISIBLE_DEVICES=`（空值）在本机**不能**禁用 CUDA，必须 `-1`；而真 CPU 上 `train_ce.py`
  会在首步段错误（rc=139），所以"CPU 冒烟"这条路在本机不成立，只能用 py_compile + GPU 小样。
- 用 Python heredoc 往 bash 脚本里写 `\` + 换行会被吞成空格（bash 续行丢失）；heredoc 里的 `0x00`
  会变成真 NUL 字节让 Python 语法检查失败。写脚本一律避免反斜杠转义。

## 5o. Phase 16 — 研究重启：域宏选择、多正例损失、多证据监督（2026-11-10 晨）

导师式备忘（`docs/RESEARCH_PLAN_NEXT.md` 的来源）指出四条代码缺口：checkpoint 用全验证查询直接平均且
训练不按域平衡；`logsumexp(all) - logsumexp(pos)` 让一篇容易正例主导损失；训练丢掉了
"top-100 无正例"的 140 条 query；只用标题会删掉时间范围/排除/比较条件。对应的实验阶梯 0–6 已落成脚本：

- `train_ce.py` 新增（默认行为不变）：`--loss possum|allpos|ndcg`、`--val-metric mean|domain_macro`、
  `--sampler uniform|sqrt`（P(域) ∝ sqrt(n)）、`--step-targets/--step-lambda`、`--q-mode title|body|full|ctxt|summ`
  （共用词预算）、`--eval-qvariants`、`--eval-win-limit`、`--init-from`、`--dump-val-qids/--val-qids`、
  `--dump-val-scores`、`--report-steps`。
- 损失对照（12 候选、3 正 9 负）：一篇正例在前时 possum 0.0541 / allpos 3.1083 / ndcg 0.1958
  三正例全在前时 possum 0.0511 / allpos 1.4587 / ndcg 0.0181 —— possum 几乎不区分，坐实"容易正例主导"。
- 约束丢失统计（13 域 1,730 条 query，脚本 `scripts/build_query_constraint_stats.py`，正则写在文件头）：年份只在正文 247 条、时间关系 328、排除 282、比较 243；
  **至少一类线索被标题化丢掉 775 条 = 44.8%**（dev 上 241/519 = 46.4%；逐 query 标记在 `results/QUERY_CONSTRAINT_FLAGS.json`）。
- 分步监督的数据半：13 域 1,121 query / 2,762 步；现有训练组覆盖 **2,321/2,547 = 91.1%** 的步骤
  （补全组 `results/ce_train_groups_bf_*` 把 140 条 query、294 篇 gold 注入后为 91.2%/98.3% query 覆盖）。
- 多证据诊断（`scripts/dev_multievidence.py`，519 dev query）：平均 2.34 步、3.44 篇相关，
  78.8% 的 query ≥2 步、87.9% ≥2 篇相关；dense → zs CE → SFT 的"top-10 里相关文档数"从
  1.03 → 1.11 → 1.21，"≥2 篇"从 31.8% → 35.6% → 38.3%，而"步骤可检索"始终 96.3%
  —— 瓶颈不是召回，是"只找最容易的那一篇"，正是 `L_step` 要压的失效模式。
- 队列（detached，等 GPU 空闲）：四臂损失/采样 → 胜者上的五表示消融 → 步骤监督 `sx_l03/sx_l10`
  → 约束保真（`cx_qv`）与补全组（`cx_bf`）→ 最终 dev 只跑一次（`ops/final_eval.sh`）。
  日志 `logs/loss_exp.log`、`logs/stage2.log`、`logs/step_exp.log`、`logs/stage3.log`。

### 5o.1 增益落在哪类问题上（stepsplit 分析，05:5x）

`scripts/eval_stepsplit.py` 按 dev query 的官方证据步数切分（qrels_steps_dev 的 `_stepK` 行），
算两系统在各分区上的域宏 nDCG@10 与配对 bootstrap（域聚类，2000 次）：

| partition | queries | zero-shot CE | frozen SFT | delta | 95% CI |
|---|---:|---:|---:|---:|---|
| 1 step | 75 | 0.3185 | 0.3896 | +0.0711 | [+0.0244, +0.1056] |
| 2 steps | 151 | 0.3857 | 0.4058 | +0.0200 | [-0.0033, +0.0463] |
| 3+ steps | 258 | 0.3196 | 0.3511 | +0.0315 | [+0.0084, +0.0571] |
| no steps | 35 | 0.1579 | 0.1705 | +0.0126 | [+0.0030, +0.0524] |

合计 519 条，与官方一致。结论（写进 PAPER_4PAGE §5.2b）：现有目标函数买的是"单篇选得更好"，
**没有教出多证据装配**——151 条两步题的增益落在噪声里。这既是 `L_step` 的目标，也是它的判据：
必须让两步分区动，而不只是总均值动。

### 5o.2 损失与表示两阶梯的结果（09:1x–09:2x）

**损失四臂**（同一 942 组、同一 45 条固定 val，域宏 nDCG@10，best epoch）：

| 目标函数 / 采样 | best epoch | val 域宏 | plain mean |
|---|---:|---:|---:|
| 当前 logsumexp-正例（possum） | 2 | **0.2865** | 0.2801 |
| 全正例平均（allpos） | 2 | 0.2809 | 0.2754 |
| nDCG 加权（LambdaLoss 式） | 2 | 0.2764 | 0.2743 |
| possum + √域 采样（sqrt） | 2 | 0.2912 | 0.2871 |
| 零样本参考 | — | 0.2730 | — |

按域宏选点与按 plain mean 选点在**四个臂上完全一致**（都是 epoch 2），即官方聚合
口径不影响 checkpoint 选择；影响选点的是 45 条 val 自身的分辨率（±0.03）。

预登记规则：胜者须比当前损失高 **0.010** 以上，否则保留当前损失。sqrt 只高
0.0047 → `results/LOSS_WINNER.json` 记 `forced: {from: lx_sqrt, to: lx_possum,
margin: 0.004692}`。机制旁证：sqrt 把「前十至少一篇相关」的 query 比例从
93.3% 抬到 **95.6%**（四臂中唯一），但 nDCG 没动——域平衡让容易的判断更容易，
没让难的判断变对。allpos 与 possum 的配对检验：Δ+0.0047，query CI
[-0.0164,+0.0285]，零分 query 数两臂完全相同（各 16/45）。

**表示五臂**（同一个 lx_possum best checkpoint，只评不训，64 词预算）：

| 表示 | val 域宏 |
|---|---:|
| full（标题+正文拼接） | **0.2869** |
| title | 0.2827 |
| ctxt（约束保真，标题+线索句） | 0.2744 |
| body（定长正文窗） | 0.2703 |
| summ（词频摘要） | 0.2614 |

45 条 val 的分辨率是 ±0.03（同 checkpoint 对零样本的配对 CI [-0.0317,+0.0571]），
full 与 title 的 0.0042 差距在噪声内，所以表示臂的胜负交给 dev（stage3 会训练
full 与 title 两个臂并用 519 条配对 bootstrap 定胜负）。ctxt 在 val 上垫底，
与 5o.1 的子集发现一致：约束句不是当前排序器缺的东西，只有在 dev 的 lost 子集
上求出增益才算数。

### 5o.3 两次队列事故（本日最有价值的工程教训）

1. **heredoc 绑错对象导致整条链静默停摆**（09:08–09:14）。`queue_stage2.sh` 里写
   `$PY - 2>&1 | tee -a "$LOG" <<'PY'`：bash 把 heredoc 绑到管道**最后一个**命令
   `tee`，于是 tee 把整段 Python 源码写进日志，而 `python -` 转去读脚本自身的
   stdin，永久阻塞。表象只是「日志停住」，实际四条下游队列都在等它写出的
   `QVARIANT_WINNER.json`。定位办法：`Get-CimInstance Win32_Process -Filter
   "Name='python.exe'"` 看到 `python.exe -` 且父进程是队列脚本的 bash。修法：
   `$PY - 2>&1 <<'PY' | tee -a "$LOG"`（heredoc 写在命令本体后、管道之前）。
2. **队列脚本引用了不存在的参数**（09:23–09:28）。`queue_step_exp.sh` 传
   `--report-steps`，但 `train_ce.py` 从未定义该参数（诊断是给了 `--step-targets`
   就自动算），两个步骤臂各在 30 秒内 `rc=2` 死掉，而队列照旧打印
   `=== step experiment done`——下游 stage3 会据此认为步骤监督已经做完。修法：
   给 `train_ce.py` 加一个真正的兼容 no-op 参数（`--help` 里可见），轮转
   `logs/step_exp.log` 让 stage3 重新等标记，再重拉两条队列。

**通用规则**：队列脚本改动后必须 `bash -n` + 用一个臂的 `--help` 或 30 秒小跑
验证参数名；队列启动后 1 分钟内要检查 `logs/<arm>.log` 有没有 `rc=` 失败，
不能只看 `START`/标记行。heredoc 在 bash 里永远属于「紧跟它的那个命令」，
跨管道时位置决定了谁读到脚本。

### 5o.4 分步监督（主线创新 A）的判定：负结果（18:0x）

**做法**：`L_step = -(1/m) Σ_j log Σ_{d∈P_j} p(d)`，`L = L_list + λ L_step`，
训练组内每步至少一篇正例才计分（覆盖率 2,518/2,762 = 91.2%）；λ ∈ {0.3, 1.0}，
两臂除该项外与 `lx_possum` 完全同参（942 groups、2 epoch、seed 17、固定 45 条 val）。

**训练内 45 条 val（域宏）**：sx_l03 0.2882（e2）、sx_l10 0.2913（e2），
都高于当前目标函数的 0.2865——λ=1.0 甚至是当天六臂里 val 最高的一支。

**dev（519 条、域宏、z 1:1 冻结融合）**：

| 系统 | 域宏 | Δ vs frozen | 域聚类 CI | 步骤覆盖率 top10 | ≥2 篇进 top10 |
|---|---|---|---|---|---|
| 冻结 sft_e2bc | 0.3581 | — | — | 59.45% | 38.31% |
| sx_l03 (λ=0.3) | 0.3442 | −0.0139 | [−0.0348,+0.0044] | 55.58% | 37.95% |
| sx_l10 (λ=1.0) | 0.3438 | −0.0143 | [−0.0351,+0.0037] | 56.00% | 37.88% |

分步分区里连"两步题"这一预登记检查点也在掉（0.4058→0.3910 / 0.3889，CI 含 0）；
四个分区（1 步 / 2 步 / 3+ 步 / 无步骤）全为负。机制指标同向下降，
即**损失没有实现它要优化的量**。

**为什么**：① 每步正例大多就是 listwise 项已经覆盖的同一批文档（平均 3.44 篇相关 /
2.34 步，一步很少带来"另一篇"）；② 步内 log-sum-exp 被"最难检索的那一步"主导，
梯度去追候选里位置很深的文档，而不是组装答案。数据侧已排除：2,873/2,873 的
步骤正例都是 query 级正例（脚本核验，100%）。

**最重要的教训**：45 条 val 上 λ=1.0 是六臂第一、dev 上却是最后一名。
**训练内划分在小样本下不是弱信号，而是错误信号**——这正是预注册"融合形式先冻结、
不在 dev 上选点"的价值所在。按预注册判据（+0.005 且域 CI 不跨 0）记为
**未证实且方向为负**，不再扫 λ（在 45 条上再扫只会买到噪声）。

产物：`results/STEP_ARM_sx_l03.md`、`results/STEP_ARM_sx_l10.md`、
`results/aggregate_sub_sx_l0{3,10}_dev.json`、`results/boot_dom_sx_l0{3,10}_vs_frozen.json`、
`results/MULTIEVIDENCE_sx_l0{3,10}.json`。

## 5p. Phase 17 — dev 端结账：三次 val 选点全错、一次崩溃、一次睡眠谋杀（2026-11-11 → 11-12）

阶梯 3/4/5 的三个候选（`lx_possum`、`cx_qv`、`cx_bf`）在 11-11 夜里到 11-12 凌晨跑完 13 域
dev。本小节把「val 上选出来的东西在 dev 上还算不算」一次性结清，并记两起工程事故
（5p.6 的睡眠谋杀最贵：11 小时 GPU 空转）。

### 5p.1 约束保真的 64 词全问题（`cx_qv`）：−0.0138

配置：训练池不变（1,071 groups / 942 训练组），`--q-mode full --q-cap-words 64`（标题 +
正文前 64 词），其余与 `lx_possum` 完全同参（2 epoch、seed 17、possum 损失）。

| 系统 | dev 域宏 | Δ vs frozen | 查询聚类 CI | 域聚类 CI | 好/坏 | val |
|---|---:|---:|---|---|---:|---:|
| 冻结 sft_e2bc | 0.3581 | — | — | — | — | — |
| `cx_qv` | 0.3443 | −0.0138 | [−0.0272, −0.0005] | [−0.0338, +0.0052] | 124/159 | **0.2920** |

0.2920 是 11-10 当天七臂在 45 条 val 上的最高分（比 base 0.2865 高 0.0055），dev 上却低
0.0138——**第三次同向的选点错误**（前两次见 §5o.4 与 §5o.2 的 sqrt 采样）。

子集机制检验（`results/CONSTRAINT_SUBSET_cx_qv.json`：241 条"线索只在正文"的 lost 与
278 条 kept，域宏 nDCG@10）：

| 系统 | lost (241) | kept (278) |
|---|---:|---:|
| dense（一阶段） | 0.3044 | 0.2892 |
| 零样本 CE | 0.3175 | 0.3272 |
| 冻结 SFT | 0.3656 | 0.3518 |
| `cx_qv` | 0.3566 | 0.3296 |

全问题臂在 lost 上只丢 0.0090、在 kept 上丢 0.0222：**额外正文确实带了信号**（相对零样本，
lost 上 +0.0391，冻结是 +0.0481），但代价摊在 278 条本来就被短标题问得更好的 query 上。
换句话说"约束保真"是对的方向、错的剂量——把精排器的输入从干净短句换成 64 词长句，
在 46% 的 query 上换来的是净损失。

### 5p.2 补全 140 条无正例 query（`cx_bf`）：同样的 −0.0138，但两个口径结论相反

配置：训练池 1,071 → 1,211 groups（补回 140 条 dense top-100 无正例的训练 query、注入
294 篇官方正例；去掉 val 划分后训练侧 942 → 1,065），`--q-mode title`，其余同参。

| 系统 | dev 域宏 | topic-macro | Δ vs frozen | 查询聚类 CI | 域聚类 CI | 好/坏 |
|---|---:|---:|---:|---|---|---:|
| 冻结 | 0.3581 | 0.3644 | — | — | — | — |
| `cx_bf` | 0.3443 | **0.3711** | −0.0138 | [−0.0288, +0.0010] | [−0.0311, +0.0012] | 149/136 |

**口径警告（本日最该记住的一条）**：按 topic-macro 报，这一臂是 **+0.0067 的胜利**
（0.3711 vs 0.3644）；按官方 domain-macro 它是 −0.0138。同一份 run、两个口径、结论相反。
这正是 §5.3 口径更正的价值：补回的 140 条训练 query 集中在少数域，它的 topic-macro 增益
被 topic 平均放大、被域平均抵消。**任何"在 topic-macro 上赢了"的说法在这份 dev 上必须用
domain-macro 重算。**

机制：294 篇补回的文档按构造是**一阶段根本检索不到**的（dense top-100 里没有它们），
训练 CE 把它们排高等于在学一个推理时永远不会发生的判断。要让它变现必须先改一阶段，而
§5.4 的十个检索侧探针都没做到。子集上两条曲线同时掉（lost 0.3471 / kept 0.3371）。

### 5p.3 stage3 的 val 配对 bootstrap 崩了，而且崩得有信息

`scripts/compare_val_arms.py:58` 抛 `KeyError: '14418_700'`：`results/val_scores_cx_bf_e2.json`
记了 46 条 query，而 `lx_possum` / `cx_qv` 都是同一批 45 条，配对 bootstrap 按 key 对齐
直接炸。46 vs 45 的来源是补全组换了训练池，val 划分跟着变（1,065 组 vs 942 组的划分）。

**dev 全量评测不受影响**（519 条同一集合，bootstrap 走 `ops/final_eval.sh` 的域聚类路径），
但这条崩溃暴露了一个此前没说透的事实：**三个 val 数字（0.2865 / 0.2920 / 0.2533）不是
严格同一把尺子**，0.2533 里掺着"换了 val 集"这一项。这也解释了 `cx_bf` 的 val（0.2533）
为何比 dev（0.3443）低那么多——它的 val 集与另两臂不可比。

`results/STAGE3_WINNER.json` 仍记 `selected: lx_possum`（base 0.2865 vs cx_qv 0.2920，
差 0.0056 落在 0.010 并列带内 → 按预注册保留 baseline）。事后看，这是本轮阶梯里唯一一次
"保守选择"救回了正确结果：dev 上 cx_qv 真的更差。

### 5p.4 `lx_possum` 的 dev：0.3470，−0.0111——第五个"val 选出来的输家"，而且它是 base

这是整个阶梯里唯一没被评过的臂，也是 stage3 按预注册并列带保留下来的 base
（`results/ce_ft/lx_possum/best`，11-10 06:27 训好）。13:29 重跑、15:19 结束，13 域全过
（`submission_validation_lx_possum.json`：0 errors，去平后重新校验）。

| 口径 | 值 |
|---|---|
| dev domain-macro nDCG@10 | **0.3469703** |
| dev topic-macro（仅供参考） | 0.3589602 |
| vs zero-shot z1:1（0.3249） | +0.02206，域聚类 CI [+0.00732, +0.03496]，p_gt_0 = 0.9982，179 好 / 134 坏 |
| vs 冻结 sft_e2bc（0.3581） | **−0.01113**，query CI [−0.02562, +0.00354]，domain CI [−0.02651, +0.00568]，p_gt_0 = 0.069，132 好 / 167 坏 |
| 多证据（n=519） | multi@10 0.3608、step_cov 57.56%、step_recall 96.33%、gold_found 98.61% |

**排除了两个最容易的误读。**
- 不是 checkpoint 选错：val 三个 epoch 的 domain-macro 是 0.2730 / 0.2790 / **0.2865**，
  epoch 2 在 plain（0.2606/0.2745/0.2801）与 domain-macro 两个口径上都是第一，`best` 就是 epoch2。
  但要点是**冻结臂的 `best` 是它自己的 epoch1**（md5 核对：`v2m3_pool8_e2/best` =
  `ff2b4ab8…` = `v2m3_pool8_e2/epoch1`；`lx_possum/best` = `64da2308…` = 自己的 epoch2）。
  两次都按"各自 val 最优"取，却选中了不同 epoch——epoch 选择本身就是 ±0.01 量级的噪声源。
- 不是训练集不同：所有臂 `train_groups` 都是 942（val 45 query / 129 组被切掉），val 域分布
  也完全一致（bitcoin 4、iota 1、quant 2……）。

**剩下两项差异没能隔离。**
1. 代码路径：`lx_possum` 走的是 Phase 17 的显式 possum 实现 + `--dump-val-scores` 那条路，
   v2m3_* 系列是旧路径。逻辑上应该等价，但没有逐 step 数值对齐过。
2. 负例池重采样：`data_loaded.docs_loaded.history` 在冻结臂是 12,489，`lx_possum` 是 14,734，
   s23 是 14,778、s29 是 14,650。**因此"seed 离散度 0.0029"其实是"重训 + 池重采样"的带**，
   不是纯 seed 噪声；带的范围是 0.3527–0.3581，而 `lx_possum` 落在带外下方 0.006。

**结论（已写进论文 §5.5 与 §7）。**
- 五个 dev 级候选（cx_qv −0.0138、cx_bf −0.0138、sx_l03 −0.0139、sx_l10 −0.0143、
  lx_possum −0.0111）全部负，方向完全一致，幅度 0.011–0.014。val 阶梯（0.2764–0.2912）
  对 dev 的排序信息量约为零，这不是"某个臂运气差"，而是这把尺子的分辨率问题（S7）。
- 唯一"救回正确结果"的是 stage3 的预注册并列带：0.0056 的 val 差 < 0.010，按规则保留 base。
  但 base 重训后仍比冻结低 0.0111，所以 **0.3581 只能作为"提交件本身的属性"报告**，
  不能声称"该方法可复现地达到 0.358"。论文用"重训带 0.347–0.358，冻结件在带顶"表述。
- 对提交件的直接影响：无。提交用的一直是冻结的 `sft_e2bc`，`lx_possum` 只是补测，
  不参与 submission。

### 5p.5 两起工程事故：睡眠谋杀 CUDA，和 PowerShell 的 uint32 陷阱

**事故 A：02:18:13 睡眠 7 秒，11 小时白跑。**

- 现象：`bash-4232`（`ops/final_eval.sh lx_possum`）02:01:08 启动，02:06 日志写到
  `logs/final_lx_possum_bitcoin.log` 的 "20/30 queries"，此后到 13:27 一行不动。
  进程 26928 一直活着，GPU 86% / 3,192 MiB、CPU 累计只有 243 秒
  （线程 `Wait/UserRequest`），`runs/rerank_lx_possum_bitcoin_dev.trec` 从未生成。
- 根因：`Get-WinEvent` 的 Kernel-Power **Id=42/107** 显示 2026-11-12 02:18:13 进入睡眠、
  02:18:20 唤醒（只睡了 7 秒）。睡眠会销毁 CUDA 上下文，而进程不抛异常，
  于是在 GPU 上空转。`LastBootUpTime` 仍是 11-05 13:17，所以不是崩溃、不是断电。
- 排除的假设（不要再重走）：
  - 不是文档过大：genealogy 语料最大 4.95 M 字符 ≈ 6.5k 窗口，最贵 query 约 4,396 窗口，
    35 条 query 合计 21,186 窗口；11-11 三个臂用同一脚本跑 genealogy 只要
    388.5 / 410.5 / 341.7 秒（`mean_ms_per_query` ≈ 8.3–9.4 s）。
  - 不是进度打印丢缓冲：`scripts/run_chunk_rerank_qvar.py:132-133` 每 10 条 flush 一次。
- 处理：`job_kill bash-4232` + `Stop-Process 26928`，GPU 归零（848 MiB），
  13:29:51 重启（逐域可恢复，bitcoin/cardano/economics 直接 skip），genealogy 13:39:58 补完。

**教训**：GPU 长跑必须配"防睡眠守卫"，而且**"进程还活着 + GPU 占用高"不等于在工作**；
判活要看日志时间戳。恢复后第一件事是比对日志最后修改时间与当前时间。

**事故 B：`SetThreadExecutionState` 在 PowerShell 里被 `0x80000000` 坑死。**

`ops/keep_awake.ps1` 第一版写的是 `0x80000000 -bor 0x00000001`。PowerShell 把
`0x80000000` 解析成**负 Int32**，于是 `-bor` 得到 `-2147483647`，P/Invoke 声明是
`uint`，调用直接抛 `CannotConvertArgument`；脚本在第一轮就死掉，但 `Start-Process -Hidden`
把错误吞了，看起来像在跑。

- 正确写法：`$ES_CONTINUOUS = [uint32]2147483648`、`$ES_KEEP_ON = [uint32]2147483649`。
  调用成功后返回值是"上一次的状态"（例如 2147483648）。
- **唯一的验证方法**是 `powercfg /requests`，SYSTEM 段必须出现
  `[PROCESS] \Device\HarddiskVolume3\Windows\System32\WindowsPowerShell\v1.0\powershell.exe`。
  第一版跑着的时候该段是"无"，就是没生效。
- 该脚本不改电源方案（只持有一个当前会话的 power request），`-Hours 10` 到点自动释放，
  注释已写进文件头部，避免下次再踩。

**当前状态**：`keep_awake.ps1` 以 `-Hours 10` 运行，`powercfg /requests` 已确认持有请求；
`final_eval lx_possum` 15:19 完成；`ops/queue_cls.sh` 已在 15:10 左右接过 GPU
（hsm 169,552 docs @ ~51 doc/s，预计 18:20–18:45 全部结束）。

### 5p.6 阶梯第 6 步（dense 池化）结账：wash，以及一颗 44.5 M 字符的 tokenizer 炸弹

第 6 步原本是"换 dense encoder 的池化"，在 11-11 的预注册里被降级为**基线校正**：
它不改系统，只回答"0.2943 这个 dense 基线是不是被我选的 mean pooling 压低了"。

- 跑法：`ops/queue_cls.sh` 逐域调用 `scripts/run_dense_generic.py --tag mxbai_cls
  --pooling cls`，其余与 mean 线完全相同（mxbai-embed-large-v1、指令前缀
  "Represent this sentence for searching relevant passages: "、top-100、fp16、
  max_length 512、同一 519 条 dev query）。13 域全部重编码，hsm 用 `ops/queue_cls_hsm.sh` 补。
- 结果（`results/aggregate_dense_mxbai_cls_dev.json`，13 域 519 query）：

  | 口径 | mean pooling | CLS pooling | 差 |
  |---|---:|---:|---:|
  | domain_macro（官方） | 0.294275 | **0.295633** | +0.0014 |
  | topic_macro（旧口径） | 0.289121 | 0.282661 | -0.0064 |

  逐域 cls - mean：bitcoin -0.0084、cardano +0.0155、economics +0.0410、
  genealogy +0.0011、history -0.0130、hsm -0.0125、iota -0.0027、law +0.0136、
  monero +0.0077、politics -0.0206、quant -0.0322、travel -0.0028、workplace +0.0311。
  两个口径方向相反、幅度都在 ±0.006 以内，落在 dev 的半宽（0.007–0.014）之下：
  **池化是 wash，0.2943 不用改**，阶梯第 6 步以"无差别"结案。论文 §3 加了一句
  "Dev: 0.2943 (0.2956 with CLS pooling: a wash)"，§5.7 结尾改成"六步里五步重训
  都输、第六步（池化）是 wash"，§7 的"未做池化消融"改成只针对 CE 的窗口池化。

- **事故：hsm 15:51:07 死于 Rust 的 OOM。** 报错原文
  `memory allocation of 5616173056 bytes failed`（5.2 GiB，Rust/tokenizers），
  死在 97,344/169,552 篇的批边界上；`ops/queue_cls.sh` 记 `FAIL dense hsm`，
  随后 aggregate 因缺这一域也 `FAIL`（12 域照常落盘）。
- 根因：`data/track1_tempo/hsm/documents.jsonl` 第 98,342 行
  `hsm/4529d6f6_210275.txt` 的 `content` 有 **44,469,997 字符**（该域平均 2,433）。
  fast tokenizer 是"整串先分词、再按 max_length 截断"，所以 512 的截断救不了它：
  44 M 字符 ≈ 1,100 万 token，单条就要几 GiB。
- 修复：`scripts/run_dense_generic.py:78-88` 的 `encode()` 在进 tokenizer 前按字符预截断
  `char_cap = 100 * max_length`（512 → 51,200 字符）。正确性论证写在注释里：词表 token
  至多几十个字符，前 512 个 token 必然落在前 100×512 个字符内，所以这个截断**不改变任何
  embedding**（对已跑完的 12 域也是零影响）。hsm 重跑 54 分钟（169,552 docs @ 53.5 doc/s，
  19:33:05 完成），score + aggregate 不到 1 秒。
- 一句保留意见：mean 线的 hsm 是**旧脚本版本**跑的（meta 里 python 3.12.7、
  peak GPU 1.4 GB、num_docs 169,552、max_length 512），它的字符截断路径没有留下来；
  hsm 上 cls 比 mean 低 0.0125 可以确定不是这次补丁造成的（补丁前后向量相同），但也不能
  完全排除旧版对超长文档的截断与新版不同。该差异只影响 hsm 一域，且被 wash 结论吸收。
- 工程教训：语料里"单条超大文档"会绕过所有基于 max_length 的内存假设——**先按字符截断，
  再进 tokenizer**；另外 D: 盘只剩 3.5 GB，`emb/` 已经 16 GB（两套池化的 doc cache 各约 7 GB），
  再跑第三个 pooling 之前要先清缓存。

## 5q. External audit of 681cd80 (2026-11-13) — what held, what did not

An external reviewer read the public repository at `681cd80` and raised four
classes of complaint. Each was checked against the local artifacts; the checks
are recorded here because two of them changed published numbers.

**1. The repository did not match the experiments.** Confirmed. `ops/` was
missing entirely, `scripts/train_ce.py` had 32 of its 43 arguments, and the
documented command failed with `unrecognized arguments: --loss --sampler
--val-metric` because those flags only existed locally. Also missing: the 65
per-query score files for the five Phase-17 arms and their 100-deep rerank runs.
All of it is now committed; the fused 1000-deep runs stay out (37 MB per arm) and
are regenerable with `scripts/fuse_scores.py`.

**2. The negative result was over-attributed.** Confirmed, and this is the
substantive correction. `lx_possum` (same objective, current code path) scores
**0.3470**, `-0.0111` against the frozen artifact, so most of the `-0.0138` the
paper charged to each correction is a retraining cost. Paired domain bootstrap
against that same-code baseline:

| arm | delta vs lx_possum | query CI | domain CI | better/worse |
|---|---:|---|---|---|
| cx_qv | -0.002650 | [-0.0168, +0.0109] | [-0.0187, +0.0100] | 144/120 |
| cx_bf | -0.002636 | [-0.0121, +0.0060] | [-0.0162, +0.0101] | 142/110 |
| sx_l03 | -0.002740 | [-0.0147, +0.0079] | [-0.0154, +0.0082] | 125/99 |
| sx_l10 | -0.003127 | [-0.0144, +0.0070] | [-0.0153, +0.0072] | 131/106 |

Every interval crosses zero. The paper's Sections 5.5 and 5.7 now separate the
code-path cost from the method effect and report the four arms at
0.0026-0.0031, not 0.0138.

**3. The 96.3% "steps retrievable" figure had no depth.** Confirmed, and worse
than the reviewer knew: it was computed on `runs/sub_*_dev.trec`, which is
1000-deep, while the cross-encoder only ever sees 100 candidates. Recomputed on
`runs/rerank_sft_e2bc_*_dev.trec` (`results/MULTIEVIDENCE_sft_e2bc_depth100.json`):
**step_recall 80.70%, gold_found 88.27%, step_cov 51.39%** (96.33/98.61/59.45 at
depth 1000). So recall is a partial bottleneck at the operating depth, not a
non-issue; the abstract and Section 5.2 now give both numbers. The retrieval
floor proper stays small: 10 of 519 dev queries (1.9%) have no relevant document
anywhere in the dense top-1000.

**4. The 45-query screen was described as promoting every loser.** Confirmed as
an inaccuracy: `LOSS_WINNER.json` kept the base because `lx_sqrt` fell inside the
pre-registered 0.010 tie band, `STAGE3_WINNER.json` also kept `lx_possum`, and
`cx_bf` was trained to fix a documented defect rather than because it won on val.
Only the full-question arm was the screen's top score. The text now says the
screen cannot rank these arms (its five scores lie within 0.006) instead of
claiming it promoted all of them.

**5. Smaller corrections, all applied.**
- The year-consistency clause is a *fixed rule* that loses (0.2980 -> 0.2498 in
  7/8 anchored domains), not an "upper bound": the reordered list is not the best
  obtainable one. Renamed everywhere, including the 4-page abstract.
- Training is listwise softmax over **20-document groups** sampled from each
  query's top-100 (`1,071 groups / 21,420 pairs`), not over 100 candidates.
- Checkpoint selection is on the cross-encoder's own ordering over 45 held-out
  train queries, not on the fused system; the frozen run selected epoch 1 on the
  plain mean because it predates domain-macro logging. `docs/REPRODUCE.md` said
  otherwise and now matches the code (`--val-metric mean --val-qids ...`).
- `scripts/run_dense_generic.py` and `scripts/make_submission.py` now put a
  *content digest* of the model directory in the cache fingerprint (path alone
  let a retrained checkpoint at the same path reuse stale embeddings) and refuse
  legacy caches without a sidecar unless `--adopt-legacy` is passed.
- `scripts/validate_submission.py` refuses a run whose domain cannot be resolved
  from its name instead of falling back to a token that silently skipped every
  check (`bad_unknown_dev.trec` previously reported `0 errors`); missing
  qrels/corpus is now an error, not a warning.

**Still open.** The fused 1000-deep runs are not committed (size). No official
test queries have been released, so there is still no leaderboard number.
`docs/PAPER_DRAFT.md` (the long version) was corrected on the same points but has
not been re-read line by line against the 4-page version.

## 5r. Second external audit of fb2bf2f (2026-11-13 02:20)

Same reviewer, re-checking the audit response. It accepted the previous round
(65 score files + 65 rerank runs recompute; fusion reproduced on iota/quant; the
four paired bootstraps match independently) and raised three new items.

### 5r.1 Eval entry could reuse stale outputs

`ops/final_eval.sh` resumed on `[ -s "$out" ]`, so the same TAG with a different
checkpoint, query mode or cap silently reused the old rerank. Fixed with
`scripts/run_complete.py`: every step stamps `<out>.done.json` holding a
fingerprint over (model dir digest, params string, each input file's sha1) plus
the output's size and sha1, and is skipped only when all three still match. A
truncated output fails the sha1 check; a config change fails the fingerprint
check. `ALLOW_ADOPT=1` is the explicit migration path for pre-marker files.

### 5r.2 `--detie` wrote nothing when there were no ties

The validator only wrote the de-tied copy when `tie_topics` was non-zero, while
`final_eval.sh` scored that file unconditionally: a tie-free run left either a
missing file or a stale one from an earlier run. Now the copy is always written
(atomic `os.replace` from `.partial`, full-precision scores) and re-read. Tested
on `fused_sft_e2bc_iota_dev.trec` (ties) and `sub_sft_e2bc_iota_dev.trec` (no
ties): both produce the file, 0 errors.

### 5r.3 Coverage numbers mixed two different orderings

The 51.39% added in §5.2/§5.6 came from `MULTIEVIDENCE_*_depth100.json`, which
measures the **cross-encoder alone** (`rerank_*`, 100 rows); 59.45% measures the
**fused submission** (`sub_*`). Closing the gap by "depth" was wrong. Control
experiment: truncating the fused run to 100 rows per query leaves its top ten
unchanged, so `step_cov` stays 59.45% and nDCG 0.3581 while `step_recall` drops
96.3% -> 80.7% (`results/MULTIEVIDENCE_sft_e2bc_sub100.json`). Depth moves
recall only. Also the denominator is all annotated steps with a positive, not
"reachable steps". Paper text rewritten accordingly; definitions collected in
`results/MULTIEVIDENCE_NOTE.md`.

### 5r.4 Smaller items

* `make_submission.py` dense fingerprint now includes a corpus digest (size,
  mtime plus head/tail MiB sha1 of `documents.jsonl`), so an in-place corpus edit
  invalidates the cached ranking.
* `validate_submission.py`: `examples` is now required (unknown/missing topics
  cannot be checked without it) and `qrels` is optional, which is what a test
  split needs; the official `format_checker.py` already treats both as optional.
* `docs/REPRODUCE.md` §2 gained the train-split dense pass that
  `build_ce_train_groups.py` needs, and says the published training groups make
  that pass optional.
* `results/CHECKPOINTS.md` now states plainly that the six checkpoints are not
  published, gives their md5s, and explains what can be re-derived without them.
* `TEMPORAL_CEILING.md` retitled: fixed rule failing, not an upper bound.

### 5r.5 Still open

No weights are published (6 x 1,082 MB); the cheapest closure is releasing
`v2m3_pool8_e2`, the artifact every headline number uses. No official test
queries yet.

---

## 5s. Why a better cross-encoder does not improve the fused system (2026-11-13)

This is the question the project pivots on after the external audit, and it is
now the intended core of the SemEval system paper: *if the cross-encoder ranks
better on its own, why does the fused system not improve?*

### 5s.1 The observation

Same 100-candidate pool, same fusion recipe (`z` per query over the common
candidates, `w = 1`), domain-macro nDCG@10 on 519 dev queries:

| arm | CE alone (top-10 over 100) | fused, w = 1 | Δ |
|---|---|---|---|
| `lx_possum` | 0.2756 | 0.3470 | — |
| `cx_qv` | 0.3153 | 0.3443 | CE +0.0397, fused −0.0027 |

Both numbers reproduce the external audit's recomputation. The rest of this
section is `scripts/ce_fusion_probe.py` (CPU only) and its output
`results/CE_FUSION_PROBE.json`; per-query scores in `results/probe_scores/`.

### 5s.2 The fusion weight is model-specific

Sweeping `w` in `z_dense + w * z_ce`:

| w | 0 | .25 | .5 | .75 | 1 | 1.25 | 1.5 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lx_possum` | .2943 | .3179 | .3309 | .3412 | **.3470** | .3439 | .3432 | .3395 | .3366 | .3325 | .3147 |
| `cx_qv` | .2943 | .3204 | .3370 | .3408 | .3443 | .3509 | .3532 | **.3546** | .3487 | .3475 | .3429 |

`lx_possum` peaks at the frozen weight; `cx_qv` peaks at 2.0, and moving from
1.0 to 2.0 is worth **+0.0103**, paired bootstrap CI [+0.0012, +0.0201]
(`results/boot_probe_cxqv_w2_vs_cxqv_w1.json`). So part of the missing gain is
not lost, it is *under-weighted*: a stronger CE wants a larger share of the
fused score, and the frozen 1:1 was calibrated on a weaker one.

At its own optimum `cx_qv` still does not beat the frozen artifact
(0.3546 against 0.3581, Δ −0.0035, CI [−0.0183, +0.0118]), but it does move
ahead of the same-recipe retrain (`lx_possum` 0.3470, Δ +0.0077, CI
[−0.0080, +0.0228]).

### 5s.3 Ordering versus score shape: rank-preserving transplant

Keep one model's *ordering* of the 100 candidates but give it the other
model's *z-profile* (the sorted z-scores assigned by rank):

| ordering | profile | w = 1 | w = 2 |
|---|---|---|---|
| `cx_qv` | `cx_qv` (own) | .3443 | .3546 |
| `cx_qv` | `lx_possum` | **.3482** | .3517 |
| `lx_possum` | `lx_possum` (own) | .3470 | .3395 |
| `lx_possum` | `cx_qv` | .3403 | .3481 |

Two readings. (i) At w = 1 `cx_qv`'s better ordering is worth at most
+0.0012 over the incumbent's fused list — the ordering gain is almost entirely
absorbed. (ii) Its *profile* is actively costly: swapping it onto the
incumbent's ordering drops 0.3470 → 0.3403 (−0.0067). `cx_qv` puts its top
z-score a little lower (3.049 against 3.095) and its mid-rank mass a little
higher (z at rank 10: 1.264 against 1.184), so under equal weights the dense
component wins the contested top slots.

### 5s.4 Calibration-free control

Rank fusion (RRF, k = 60) throws the scores away and uses only the two orders:
`lx_possum` 0.3393, `cx_qv` 0.3462 — the better CE is **+0.0069** ahead, about
a sixth of its CE-only gain. So the ordering does carry signal; score-level
fusion at w = 1 (Δ −0.0027) loses it to calibration.

### 5s.5 Where the 0.04 CE gain goes, query by query

Per-query Δ(CE-only) against Δ(fused, w = 1), 519 queries:

| group | n | mean Δ CE-only | mean Δ fused w=1 | sum |
|---|---|---|---|---|
| `cx_qv` CE better (Δ > 0.01) | 185 | **+0.1985** | **+0.0458** | +8.48 |
| `cx_qv` CE score unchanged (|Δ| ≤ 0.01) | 221 | +0.0001 | **−0.0116** | −2.55 |
| `cx_qv` CE worse (Δ < −0.01) | 113 | −0.1398 | −0.0482 | −5.45 |

Regression of Δfused on ΔCE: slope **0.302** at w = 1, **0.377** at w = 2 — a
CE improvement reaches the fused top ten attenuated roughly threefold. The
three flows nearly cancel: net +0.0009 per query, which is the −0.0027 macro
difference after domains are re-weighted.

The middle row is the one worth remembering: on 221 queries the better CE's own
nDCG@10 does not move, and the fused system still loses 0.0116. Their CE top-10
overlap with the reference arm is only 0.61, so the CE list does move — what
does not move is its score on gold, which is what the single-stage metric sees.
The clean isolation is the transplant: at w = 1, `cx_qv` ordering over
`lx_possum` profile scores 0.3482 (+0.0012 over `lx_possum` itself) while
`lx_possum` ordering over `cx_qv` profile scores 0.3403 (−0.0067). At the frozen
weight the fused list follows the score profile more than the better ordering,
which is a calibration effect rather than an ordering one.

Fused top-ten overlap: 74% with the dense top ten, 55–60% with the CE top ten
for both arms, i.e. the fused list is dense-anchored, which is why CE changes
arrive attenuated.

### 5s.6 Selection implication

Across the six arms measured on dev (`MULTIEVIDENCE_*_depth100.json` for
CE-only, `aggregate_sub_*_dev.json` for fused):

| arm | CE-only | fused | CE rank | fused rank |
|---|---|---|---|---|
| `sft_e2bc` (frozen) | 0.2823 | **0.3581** | 2 | 1 |
| `lx_possum` | 0.2756 | 0.3470 | 3 | 2 |
| `cx_qv` | **0.3153** | 0.3443 | 1 | 3 |
| `cx_bf` | 0.2529 | 0.3443 | 6 | 4 |
| `sx_l03` | 0.2739 | 0.3442 | 4 | 5 |
| `sx_l10` | 0.2720 | 0.3438 | 5 | 6 |

The best CE-only arm is third fused; the frozen artifact is second on CE-only
and first fused. Spearman across six arms is 0.66, but the two extremes invert,
which is exactly the regime where a 45-query CE-only validation screen decides
the wrong way. The pipeline in `docs/REPRODUCE.md` selects checkpoints on the
cross-encoder's own val nDCG; this table says that objective is only loosely
coupled to the deployed one.

### 5s.7 The weight transfer test on 45 held-out queries

`ops/queue_val_weight.sh` rebuilt the dense top-100 for the 45 held-out training
queries (`results/val_qids_e2.json`), reranked them with both checkpoints, and
swept `w` on val (`results/VAL_WEIGHT_SWEEP.json`):

| w | 0 | .25 | .5 | .75 | 1 | 1.25 | 1.5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| `lx_possum` val | .2824 | .3119 | .3241 | **.3336** | .3311 | .3081 | .3056 | .2924 | .2798 | .2793 |
| `cx_qv` val | .2826 | .3050 | .3088 | .3104 | .3197 | **.3280** | .3167 | .3164 | .3054 | .3231 |

The optima move the right way — `lx_possum` at 0.75, `cx_qv` at 1.25 — but
45 queries cannot pin a weight: for `lx_possum` the top two weights differ by
0.0025, and applying the val argmax to dev **costs** 0.0058 (0.3412 against
0.3470 at w = 1, paired CI [−0.0117, −0.0003], i.e. a real loss). Applied the
same way, `cx_qv` reaches 0.3509, which is +0.0097 over `lx_possum` at val
weights (CI [−0.0021, +0.0212], P = 0.945) and −0.0072 against the frozen
artifact (CI [−0.0203, +0.0058]).

So the honest reading:

* the **calibration channel is established**: the transplant test and the
  221-query group do not depend on any weight-selection split;
* the **direction** of the weight shift is confirmed on held-out queries
  (`cx_qv` wants more CE than `lx_possum`), but a 45-query argmax is too noisy
  to be a method — it hurts the incumbent;
* a **calibration-free** fusion (RRF k = 60) recovers +0.0069 of the CE gain
  without any weight (0.3393 → 0.3462, CI [−0.0087, +0.0221]) but still trails
  the frozen artifact by 0.0119.

The paper claim therefore stays narrower than "we recover the gain": *in a
dense+CE system, CE-only nDCG@10 is a weak proxy for fused nDCG@10 (the best
CE-only arm is third fused, the second-best is first), roughly a third of a CE
change reaches the fused top ten, and part of the rest is score calibration
rather than ordering.* Recovering it needs either a tuning split much larger
than 45 queries or a calibration-free fusion, and neither closes the gap to the
frozen artifact on dev.

Next experiment, still cheap: fused-val checkpoint selection for one arm pair
(three epochs of `cx_qv`, rerank val, compare "pick epoch by CE-only val" with
"pick epoch by fused val", then dev-score the two picks). That is the part of
the selection story that is currently inferred from six arms rather than
demonstrated.

## 5t. Goal round 1 — the screen and the deployment read different token budgets (2026-11-13 03:15 →)

The publication goal (m01842) turned the selection question from a paragraph in
5s.7 into a measurement. Two things were done in this round: the mechanism was
written into the paper (5.8) with a new error analysis (8) and conclusion (9),
and the checkpoint screen was measured against the fused metric on the 45
held-out training queries.

### 5t.1 What the screen actually optimises

`scripts/train_ce.py` selects `best/` on `val_ndcg10_domain_macro` (or the plain
mean for the frozen run, which predates that key), computed inside `validate()`
on 45 held-out train queries. Until this round the assumption was that this is
"the cross-encoder's quality". It is not, for two independent reasons:

1. **It is a single-stage metric.** The submitted system fuses dense and CE
   scores; 5s showed that a CE-only improvement does not survive the fusion
   (anchor / attenuation / calibration).
2. **It is measured at a different token budget than deployment.** Every
   experimental arm (`ops/queue_loss_exp.sh:33`, `queue_seeds.sh:33`,
   `queue_stage2.sh:97`, `queue_stage3.sh:22`, `queue_step_exp.sh:27`,
   `queue_sx10.sh:35`) passes `--max-len 320` with the default `--win 250`, and
   `validate()` tokenises with `max_length=args.max_len` — so the pair is
   truncated to 320 tokens. Deployment re-ranks with
   `max_length = q_tokens + win + 4 = 256 + 250 + 4 = 510`
   (`scripts/run_chunk_rerank_qvar.py`, before this round). Same checkpoint, same
   candidate list, two different readings of each document.

### 5t.2 Re-scoring the screen at the deployment budget

`scripts/run_chunk_rerank_qvar.py` gained `--pair-max-len` (0 = the old
`q_tokens + win + 4`), so the two budgets can be compared on identical
candidates. `scripts/val_epoch_select.py` then reports, per epoch, CE-only
nDCG@10 and the fused nDCG@10 at w = 1 on the same 45 queries.
`ops/queue_val_epochs.sh` produces `runs/rerank_<tag>_<d>_val.trec`.

lx_possum, 45 queries, deployment budget (510):

| epoch | CE-only | fused w=1 | top-10 ∩ dense-CE |
|---|---|---|---|
| 1 | **0.2462** | **0.3325** | 0.575 |
| 2 | 0.2396 | 0.3311 | 0.564 |

Training-time screen (320): ep0 0.2606/0.2730, ep1 0.2745/0.2790, ep2 0.2801/0.2865
(plain / domain-macro), i.e. the screen kept **epoch 2**. At the deployment
budget both CE-only and fused prefer **epoch 1**. The cardano slice shows the
same reversal directly: at 320, ep2 0.1799 > ep1 0.1717; at 510 both 0.2020.
So the epoch choice is a property of the truncation budget, not of the data.

This does not say which epoch scores better on dev — that is
`ops/queue_lx_ep1_dev.sh`, and the answer arrived at 05:18: **ep1 0.3428,
ep2 0.3470**, i.e. the epoch the deployment-budget screen prefers is 0.0042
*worse* on dev (domain CI [-0.0114, +0.0022], 42 better / 80 worse / 397 tied).
So the 320/510 difference is real but it is not the reason anything failed:
both val screens are underpowered at 45 queries, and the original screen's
choice happened to be the better one. The lesson is the one 5s.5 already
carries - a 45-query screen cannot rank arms this close - plus a new footnote:
the screen's *preprocessing* is not the deployment's, so its numbers should not
be re-used as system numbers even when the ordering survives.

### 5t.3 A correction to 5s.5

5s.5 grouped queries by "CE top-10 unchanged". That is not what was computed:
the group of 221 queries is those where **CE nDCG@10 itself does not move**
(|Δ| ≤ 0.01), and their top-10 lists overlap with the reference only 0.609 on
average (recomputed 2026-11-13). The paper and 5s now say "CE score unchanged",
and 5s.5 carries the transplant numbers (cx ordering + lx profile 0.3482,
lx ordering + cx profile 0.3403) as the direct isolation of order from profile.

### 5t.4 The held-out replication, and what it does not show

`val_epoch_select.py` on all four checkpoints (45 held-out train queries,
deployment budget, same candidates, fused at w = 1):

| checkpoint | CE-only | fused w=1 | top-10 overlap |
|---|---:|---:|---:|
| lx_possum ep1 | 0.2462 | **0.3325** | 0.575 |
| lx_possum ep2 | 0.2396 | 0.3311 | 0.564 |
| cx_qv ep1 | **0.2893** | 0.3197 | 0.596 |
| cx_qv ep2 | 0.2882 | 0.3151 | 0.594 |

Two readings, one of which is a negative result for the hypothesis this round
started from:

- **Across models the inversion replicates without dev.** cx_qv is 0.043 better
  alone and 0.013 worse fused, on queries that played no part in the dev
  analysis. This is the cleanest form of the 5.8 claim we have: same candidates,
  same fusion, same 45 queries, opposite order.
- **Within a model the two criteria agree on the epoch.** Both prefer epoch 1
  for lx_possum and both prefer epoch 1 for cx_qv. So substituting the fused
  metric for the CE metric would not have changed either epoch choice. The
  selection story is therefore about *which model to ship*, not about which
  epoch of one model - the epoch-level error that did occur came from the
  token-budget mismatch (5t.2), not from the single-stage metric.


### 5t.5 The dev check: the flipped epoch is worse, not better

`ops/queue_lx_ep1_dev.sh` finished at 05:18 (13 domains, 519 queries,
~90 GPU-minutes) and produced `results/aggregate_sub_lx_ep1_dev.json`:

| lx_possum checkpoint | dev domain-macro | vs frozen | vs the other epoch |
|---|---:|---:|---:|
| epoch 1 (deployment-budget pick) | 0.3428 | -0.0153 [-0.0286, -0.0008] | -0.0042 [-0.0114, +0.0022] |
| epoch 2 (original screen pick) | 0.3470 | -0.0111 [-0.0265, +0.0057] | - |

The 45-query screen at the deployment budget ranks these two the other way
round, and dev says it is wrong by 0.0042 with 397/519 topics tied. This is the
C7 lesson in its cleanest form: at this screen size the differences between
epochs of one model are below the resolution of the metric, so any argument
that rests on "the screen picked the wrong epoch" cannot be supported - and the
one we can measure went the other way. What survives is the cross-model
inversion (5t.4) and the finding that the screen's preprocessing (320) differs
from deployment's (510); the latter is a reproducibility footnote, not a
performance claim.

### 5t.6 Still open

- ~~`lx_ep1` dev evaluation~~: done in 5t.5; the deployment-budget pick is
  0.0042 worse on dev, so the reversal is confined to the 45-query screen.
- ~~cx_qv epochs~~: done in 5t.4; CE-only and fused agree on epoch 1 for both
  models, so the selection story is cross-model only.
- Page budget: PAPER_4PAGE is now ~6,100 words; the SemEval plan asks for
  ~4,500. This is the main remaining writing task.
- Weight release and official-test predictions remain blocked on the user's
  call / the organisers.

### 5u Rendering the paper: a real template, a real page count

The "4PAGE" file name had never been checked against a template. It now is:
`scripts/md2tex.py` converts `docs/PAPER_4PAGE.md` into an ACL-style LaTeX
document (`paper/main.tex`, official `acl.sty` + `acl_natbib.bst` from
acl-org/acl-style-files, numeric natbib) and MiKTeX at
`D:/miktex/miktex/bin/x64/pdflatex` builds `paper/main.pdf` in two passes.

Measured, not guessed:

| version | words | pages | body pages |
|---|---:|---:|---:|
| first build (before any trim) | 6,104 | 9 | 8 + a stray float |
| after trimming 5.5-5.8 | 5,792 | 8 | 7 + 1 page of references |

Two conversion defects were found and fixed on the way, both worth remembering:
the Markdown headings carry their own numbers ("5.2b"), so the converter emits
starred sections (otherwise LaTeX renumbers and 5.2b collapses, shifting every
later number); and wide tables overflow a single column by 260 pt, so they are
set with `table` at `footnotesize` after the number of columns was cut to fit.

Citations were the other submission-blocking gap: the body had zero `\cite`
commands. `[@key]` now maps to `\citep{...}`, the prior-work paragraph cites the
retrieve-then-rerank cross-encoder, M3-Embedding, LambdaRank/LambdaLoss, HLATR
and JPR, and the bibliography is a real `thebibliography` with 14 entries (the
last three - M3-Embedding 2402.03216, HLATR 2205.10569, JPR EMNLP 2021 - were
checked against arXiv and the ACL Anthology before adding).

Still open: 8 pages against an unknown limit. SemEval's CFP decides whether a
system paper is 4, 6 or 8 pages; until it appears, the honest target is "trim
to 6 and keep the numbers", which means cutting ~1,200 words from the analysis
section - the next writing pass. Eight overfull boxes remain (long `\texttt`
paths inside narrow columns); they are cosmetic at this stage.

## 5v. Page budget closed: 5,115 words, 6 pages body + 1 references (2026-11-13)

The paper was rendered through `scripts/md2tex.py` into `paper/main.tex`, compiled
with MiKTeX/pdflatex against the official `acl.sty` (review mode, line numbers).

- 6,104 words -> 9 pages (first render, tables overflowing).
- section 5 rewritten against the same numbers (2,862 -> ~1,900 words), abstract
  318, sections 3/4/6/7/8/9 tightened: **5,115 words -> 7 pages total = 6 body +
  1 references, 0 overfull hbox**. Body ends exactly at the bottom of page 6.
- Tables are `resizebox`-scaled to column width (5 or fewer columns) or to text
  width in `table*` (6+ columns); the previous 8 overfull warnings (up to 301 pt,
  i.e. tables spilling into the next column) are gone.
- References are real `thebibliography` entries (15), including the three verified
  lines (HLATR arXiv:2205.10569, JPR EMNLP 2021 pp. 6997-7008, M3 arXiv:2402.03216)
  and Hinton et al. 2015 for the distillation remark; every citation in the text
  resolves (no `??` in the PDF).
- Still unknown: whether RETECO's system papers are 4, 6 or 8 pages. The draft is
  honest at 6+1; if the call is 4 pages, section 5 needs a second, harder cut (to
  ~1,200 words) or a move of S5.5-S5.8 into an appendix.

