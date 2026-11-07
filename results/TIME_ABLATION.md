# 时间特征重排消融（Phase 3）

## 设计（跑之前固定，dev 只报告一次）

* **候选冻结**：dense `bge-base-en-v1.5` 的 top-100，比较对象就是这条 dense 排序。
* **特征**（全部来自 query/guidance 与文档正文，语料无时间元数据）：
  * `doc_has_year`, `doc_year_count`：文档内四位年份（1950–2029）的出现与数量；
  * `query_has_year`, `year_overlap`, `year_distance`：query 年份与文档年份的重合、最近距离；
  * `anchor_match`：官方 guidance 的 `key_time_anchors`/`query_temporal_signals` 词项在文档中的命中比例；
  * `is_temporal`：guidance 的 `is_temporal_query`。
* **对照特征**（证明不是"长度/词面重叠"在起作用）：文档长度、查询长度、词面重叠率。
* **模型**：logistic regression（sklearn，标准化，C=1.0），**只在 train split 拟合**
  （labels 来自 qrels_train），直接应用到 dev；dev 不做任何调参。
* **消融**：`dense_only`（只有 dense 分数，应严格复现 dense 排序）、`dense_time`、
  `dense_ctrl`、`dense_time_ctrl`。

## 结果（nDCG@10，候选深度 100）

| 配置 | Bitcoin dev (30 q) | Δ vs dense | Economics dev (25 q) | Δ vs dense |
| --- | --- | --- | --- | --- |
| dense（基线） | 0.2171 | — | 0.1891 | — |
| dense_only（LR，仅 dense 特征） | 0.2171 | 0.0000 | 0.1891 | 0.0000 |
| dense + time | 0.1941 | −0.0230 | 0.2123 | +0.0232 |
| dense + controls | 0.1817 | −0.0354 | 0.1621 | −0.0270 |
| dense + time + controls | 0.1737 | −0.0434 | 0.1718 | −0.0173 |

配对 bootstrap（10k，seed 0）：

| 对比 | Δ | t | 95% CI |
| --- | --- | --- | --- |
| Bitcoin dense+time | −0.0230 | −1.22 | [−0.0651, +0.0078] |
| Economics dense+time | +0.0232 | +0.90 | [−0.0254, +0.0773] |

## 解释

1. **`dense_only` 严格复现 dense 分数**（两个域都精确到 4 位），说明管线没有引入
   额外排序噪声——这是实验有效性的自检。
2. **时间特征的效果在域之间变号**：Bitcoin −0.023、Economics +0.023，两个 CI 都跨 0。
   → 目前没有证据表明"年份匹配 + 时间锚点匹配"这类内容时间特征能稳定重排。
3. **对照特征在两个域都变差**，说明长度/词面重叠特征会把 dense 已排序好的头部打乱；
   时间特征在 Economics 上的正号很可能来自与 dense 的弱相关而非真正的时间推理。
4. 与 Phase 0 诊断一致：Bitcoin gold 文档只有 12.1% 含年份、Economics 也只有少数，
   基于年份的硬/软信号本来就缺乏覆盖。

## 结论

* 以"年份/锚点"为载体的 content-time 特征，**不能**带来可靠提升；不建议按这个方向
  继续堆特征。
* 若仍要做时间模块，需要换成**语义层的时间理解**（例如判断文档讨论的事件/时期与
  query 要求的时间段是否语义一致，而不是字面年份匹配），并且必须先解决
  "gold 文档大多不含显式日期"这一覆盖问题。
* 实验规范保持：train 拟合、dev 报告、bootstrap 给区间、失败配置也记录。

## 文件

* 代码：`scripts/time_feature_rerank.py`
* run：`runs/timefeat_{dense_only,dense_time,dense_ctrl,dense_time_ctrl}_{bitcoin,economics}_dev.trec`
* 评分：`results/score_timefeat_*.json`，配置：`results/timefeat_*.json`
