p = r'D:/RETECO/docs/RESEARCH_LOG.md'
s = open(p, encoding='utf-8').read()
anchor = '## 6. Agent 协作记录'
add = '''## 5c. Phase 4：语义时间一致性可行性检验（已完成，负结果）

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

'''
assert anchor in s
s = s.replace(anchor, add + anchor)
open(p, 'w', encoding='utf-8').write(s)
print('log updated')
