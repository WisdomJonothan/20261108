# Phase 6：三项工程化（强检索器 / 低成本 pairwise / 两视图流水线）

## ① 更强的一阶段检索器（bge-large-en-v1.5）

* 文档向量：`BAAI/bge-large-en-v1.5`（335M，1024 维），fp16，CLS pooling，
  max_length 512，与 bge-base 相同的 instruction 与检索流程；
* 编码规模：Bitcoin 76,706、Economics 74,692、IOTA 9,680（一次性编码，缓存复用）；
* 文档编码耗时：Bitcoin 76,706 篇约 21 分钟、Economics 74,692 篇约 21 分钟、IOTA 9,680 篇约 3 分钟（RTX 4060，fp16，约 59 doc/s）。

| 领域/split | bge-base nDCG@10 | bge-large nDCG@10 | Δ | 95% CI | 好/差/平 |
|---|---|---|---|---|---|
| Bitcoin dev (30) | 0.2171 | 0.1569 | **−0.0602** | [−0.107, −0.022] | 3/12/15 |
| Bitcoin train (70) | 0.1550 | 0.1673 | +0.0123 | [−0.014, +0.040] | 16/12/42 |
| Economics dev (25) | 0.1891 | 0.1835 | −0.0056 | [−0.068, +0.058] | 7/8/10 |
| Economics train (58) | 0.1209 | 0.0918 | −0.0290 | [−0.062, +0.001] | 11/14/33 |
| IOTA dev (3) | 0.4712 | 0.5856 | +0.1145 | [−0.103, +0.447] | 1/1/1 |
| IOTA train (7) | 0.2131 | 0.2381 | +0.0250 | [−0.062, +0.100] | 3/1/3 |

**结论：bge-large 不是改进。** 四个大样本格子里三个变差，Bitcoin dev 显著变差
（−0.060，CI 不含 0）；Bitcoin/Economics 合并（183 query）加权 Δ = **−0.015**。
IOTA 的正号只有 3/7 条 query，不可采信。

排除技术原因：fp16 vs fp32 的文档向量余弦相似度 = 1.00000（600 篇抽样），
说明不是半精度数值问题；instruction、pooling、max_length 与 bge-base 完全一致；
recall@100 只从 0.569 降到 0.512，也不是表示崩溃。**是模型在该领域排序偏好不同**，
与"MTEB 更强 ≠ 本任务更强"一致。

## ② 低成本 pairwise：只重排 top-5

把 Round-robin 从 top-10（45 次调用/query）降到 top-5（10 次调用/query），
其余不变；输出 = 重排后的 top-5 接 dense 的第 6–100 名（其余候选不动）。

| 领域/split | dense nDCG@10 | top-5 重排 | Δ | 95% CI | 好/差/平 | 延迟 |
|---|---|---|---|---|---|---|
| Bitcoin dev (30) | 0.2171 | **0.2670** | +0.0499 | [−0.021, +0.129] | 6/5/19 | 2.7 s/query |
| Bitcoin train (70) | 0.1550 | 0.1437 | −0.0113 | [−0.032, +0.009] | 8/11/51 | 2.4 s/query |
| Economics dev (25) | 0.1891 | **0.2026** | +0.0135 | [−0.004, +0.036] | 4/3/18 | 2.7 s/query |
| Economics train (58) | 0.1209 | 0.1273 | +0.0065 | [−0.009, +0.026] | 5/6/47 | 2.6 s/query |

成本从 11.8 s/query 降到 ~2.5 s/query（约 4.6 倍），但**收益不稳定**：
Bitcoin dev +0.050 与 Bitcoin train −0.011 方向相反，说明 dev 上的大增益由少数 query 驱动，
不能当作可复现提升。四组合并后的均值约 +0.008。


### 参考：只重排 top-5 的理论天花板

| 领域/split | dense | top-5 重排天花板 | 可提升空间 | pairwise 实际 | 捕获比例 |
|---|---|---|---|---|---|
| Bitcoin dev | 0.2171 | 0.3200 | +0.1029 | 0.2670（+0.0499） | 48% |
| Bitcoin train | 0.1550 | 0.2071 | +0.0521 | 0.1437（−0.0113） | 0% |
| Economics dev | 0.1891 | 0.2343 | +0.0452 | 0.2026（+0.0135） | 30% |
| Economics train | 0.1209 | 0.1539 | +0.0331 | 0.1273（+0.0065） | 20% |

天花板 = 把 top-5 里所有 gold 提到最前、其余保持 dense 顺序。pairwise 在三格捕获了
20–48% 的空间，但 Bitcoin train 为负——再次说明其收益不稳定。

## ③ 两视图流水线（检索 full + 精排 title + RRF 融合）

流程：
1. dense（bge-base）用**完整 query**取 top-100；
2. `bge-reranker-v2-m3` 用**标题式 query**、窗口 250 重排同一批候选；
3. 两路排名做 RRF（c=60，权重 dense:CE = 2:1），输出 top-100。

| 领域/split | dense | 两视图 | Δ | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|
| Bitcoin dev (30) | 0.2171 | **0.2311** | +0.0140 | [−0.030, +0.052] | 0.75 |
| Bitcoin train (70) | 0.1550 | **0.1713** | +0.0163 | [−0.009, +0.043] | 0.90 |
| Economics dev (25) | 0.1891 | 0.1921 | +0.0030 | [−0.024, +0.032] | 0.58 |
| pooled (125) | — | — | **+0.0131** | [−0.005, +0.032] | 0.92 |

方向在三个测试单元上一致为正（这是前面几轮里少见的），但 125 条 query 下仍未达到显著。

## 结论与系统建议

1. **两视图流水线是当前最稳的可用组合**：+0.013（pooled，P=0.92），成本仅一次 CE 前向
   （~0.7 s/query，比 LLM 便宜两个数量级）。
2. **top-5 pairwise 是"质量模式"选项**：成本 2.5 s/query，dev 上有 +0.05 的亮点，但 train 反向，
   不建议作为默认。
3. 强检索器（bge-large）的结论取决于上面的表；如果它稳定优于 bge-base，则所有下游组件应
   在该候选集上重跑。

## 文件

* 检索器：`scripts/run_dense_bge.py --model .../bge-large`，run：`runs/dense_bgelarge_*.trec`
* top-5 pairwise：`scripts/run_pairwise.py --k 5`，run：`runs/pairwise5_qwen_*.trec`、
  `runs/pipe_pw5_*.trec`
* 两视图：`scripts/fuse_rrf.py`，run：`runs/twoview_*.trec`
