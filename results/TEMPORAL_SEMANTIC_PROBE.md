# 语义层时间一致性可行性检验（Phase 4）

## 问题

在**没有发布时间元数据**的语料上，能否用"语义时间一致性"给文档打分，并且这个信号能
在 dense 语义分之外提供增量？（次要做到才值得做时间重排。）

## 设计（先固定，后跑）

* 时间陈述构造（不看 qrels，只用官方 guidance）：对每条 query 用
  `query_temporal_events` + `key_time_anchors` + `query_summary` 拼成一句
  "这篇文档是关于 X 的，相关时间/条件是 Y"。
* 打分：`temporal_cos` = bge-base（与检索器同款）对时间陈述的向量与文档向量的余弦。
* 标签：qrels（gold vs non-gold），候选为 dense top-100 全部（不做负采样）。
* 指标：单分数 AUC；**query 分组的 5 折 CV** 逻辑回归的 AUC（dense vs dense+temporal）；
  **逐 query AUC** 分布与 query-cluster bootstrap CI；并列 ΔAUC 与胜率；与 dense/词面重叠的
  相关性（共线性）。
* 对照：**placebo**——把时间陈述换成另一条 query 的陈述（确定性错排），若 placebo
  与真陈述同样有效，说明信号只是主题相似性。
* 预先设定的采纳门槛（评审建议）：ΔAUC ≥ 0.05–0.08 且 CI 下界 > 0、≥2/3 query 改善、
  nDCG@10 提升 ≥0.02。达不到就不做重排。

## 结果

### Bitcoin dev（30 query，3,000 个 query-doc 对，50 个 gold）

| 分数 | AUC |
|---|---|
| dense(-rank) | 0.7528 |
| **temporal_cos（真陈述）** | **0.6121** |
| temporal_cos（placebo 错排陈述） | 0.3887 |
| 词面重叠 | 0.5270 |
| 文档长度 | 0.3982 |

* 逐 query AUC（27 条有正负样本）：temporal 均值 **0.693**，cluster bootstrap 95% CI [0.593, 0.782]；
  placebo 均值 **0.377** → 真陈述确实携带 query 相关的信息，不是纯主题相似。
* 增量：CV AUC dense 0.7388 → dense+temporal 0.7389（**Δ = +0.00005**）；
  逐 query 合并 AUC dense 0.7605 → 0.7543（**Δ = −0.0062**，胜率 29.6%，CI [−0.0136, +0.0012]）。

### Economics dev（25 query，2,500 对，有正负样本 21 条 query）

| 分数 | AUC |
|---|---|
| dense(-rank) | 0.7690 |
| temporal_cos | 0.6478 |
| 词面重叠 | 0.7134 |
| 文档长度 | 0.6755 |

* 逐 query AUC：temporal 均值 0.6247（CI [0.531, 0.717]）。
* 增量：CV AUC 0.7535 → 0.7600（**Δ = +0.0065**）；逐 query 合并 AUC 0.7434 → 0.7442
  （**Δ = +0.0008**，胜率 47.6%，CI [−0.0117, +0.0133]）。

## 判断

1. 语义时间陈述**有可测信号**（真 0.69 / placebo 0.38），但它与 dense 高度重叠：
   在 dense 之上加它，ΔAUC 只有 0.0001–0.0065，胜率 30%/48%，CI 跨 0。
2. 远低于预先设定的采纳门槛（ΔAUC ≥ 0.05）。**结论：不做时间重排**——按约定停在这里。
3. 这也解释了为什么 Phase 3 的字面年份特征同样无效：问题不在"用哪种特征"，而在于
   dense 语义分已经吃掉了这部分信息，而语料里真正独立的时间证据（显式日期）覆盖率太低
   （Bitcoin gold 仅 12.1% 含年份）。

## 与文献的一致性（调研代理）

* 文本无时间戳场景下，已发表工作（TempRALM 等）依赖**附带的 passage 时间戳**，
  一旦去掉时间戳就没有可比的 AUC 基准；
* 时间问答基准（TempQuestions、TimeQA）也都是"有知识库/时间戳"的设定；
* 也就是说，"无元数据的语义时间一致性"目前没有现成的强基线可抄，我们的负结果与
  这一现状一致。

## 评审代理提出的问题与处理

| 评审意见 | 处理 |
|---|---|
| 指导语可能泄漏 gold 内容、余弦可能只测主题 | 加 placebo（错排陈述）对照：真 0.69 vs 假 0.38，说明确有 query 特定信息 |
| dense top-100 负样本压缩方差、AUC 阈值受采样影响 | 用全部 top-100 对、不做负采样；同时报逐 query AUC 与 cluster bootstrap |
| 只看 AUC 不够 | 增加 CV 逻辑回归、ΔAUC、胜率、CI、与 dense/词面重叠的相关性 |
| 需要效应量门槛 | 事前设定 ΔAUC ≥ 0.05–0.08；实测 ≤0.0065，未达标 |
| 小样本功效不足 | 结论按"未发现可用增量"表述，不宣称"时间信号不存在" |

## 产物

* `scripts/temporal_probe.py`、`scripts/temporal_probe2.py`
* `results/temporal_probe2_{bitcoin,economics}_dev.json`、`..._bitcoin_dev_placebo.json`
* 原始候选与标签来自 `runs/dense_bge_*` + `data/track1_tempo/*/qrels_*`
