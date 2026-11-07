# Query 表示对一阶段检索的影响（Phase 5）

## 问题

Phase 1 里"标题式 query"让所有 cross-encoder 变好（+0.03~0.05）。这是否意味着
**检索器本身**也该用短 query？本次实验直接回答：在完全相同的文档向量上，只改 query
表示，比较四种视图。

## 设计

* 文档向量：复用 `emb/doc_emb_bge-base-en-v1.5_<N>.npy`（与基线完全一致，未重算）；
* 只重算 query 向量（bge-base，同一 instruction，CLS pooling，cosine）；
* 四种视图：
  * `full`：整条 query，去 HTML 标签（基线）；
  * `title`：第一个 `<p>` 之前的标题；
  * `summary`：官方 guidance 的 `query_summary`；
  * `title_summary`：标题 + 摘要。
* 领域/split：Bitcoin、Economics、IOTA × dev/train（共 6 组），top-1000 候选。
* 统计：与 `full` 的逐 query 配对 bootstrap（5,000 次）。

## 结果（nDCG@10）

| 领域/split | full | title | summary | title+summary |
|---|---|---|---|---|
| Bitcoin dev (30) | **0.2152** | 0.1543 | 0.1230 | 0.1447 |
| Bitcoin train (70) | **0.1647** | 0.1477 | 0.1545 | 0.1266 |
| Economics dev (25) | **0.1768** | 0.1059 | 0.1005 | 0.1057 |
| Economics train (58) | 0.1262 | 0.1064 | **0.1495** | 0.1199 |
| IOTA dev (3) | **0.4553** | 0.3648 | 0.2072 | 0.2937 |
| IOTA train (7) | **0.2083** | 0.1749 | 0.0823 | 0.1606 |

配对 Δ 与 95% CI（相对 full）：

| 领域/split | title | summary | title+summary |
|---|---|---|---|
| Bitcoin dev | −0.0609 [−0.125, −0.005] | −0.0922 [−0.173, −0.016] | −0.0705 [−0.134, −0.014] |
| Bitcoin train | −0.0171 [−0.059, +0.022] | −0.0103 [−0.067, +0.044] | −0.0381 [−0.079, −0.003] |
| Economics dev | −0.0709 [−0.146, −0.002] | −0.0762 [−0.188, +0.025] | −0.0710 [−0.146, −0.011] |
| Economics train | −0.0198 [−0.063, +0.020] | **+0.0233** [−0.029, +0.073] | −0.0063 [−0.044, +0.030] |
| IOTA dev | −0.0906 [−0.246, +0.179] | −0.2482 [−0.637, +0.183] | −0.1616（样本 3 条） |
| IOTA train | −0.0334 [−0.083, +0.000] | −0.1260 [−0.257, +0.003] | −0.0478 [−0.200, +0.117] |

## 结论

1. **一阶段检索用完整 query 最好**：6 组里 5 组 full 第一，唯一例外是
   Economics train 的 summary（+0.023，CI 跨 0，且同域 dev 上 summary 反而 −0.076），
   不能当作稳定收益。
2. **短 query 帮的是重排器，不是检索器**：Phase 1 的 title 收益来自"把 OOD 的长 query
   变成 CE 习惯的短 query"；检索器本来就吃全量上下文，砍掉正文会丢信息。
3. 实践含义：**两套表示分开用**——检索用 full query，精排阶段再用 title/摘要视图。
   这也是一个可写进论文的系统性观察（query view 的收益依组件而异）。

## 附：HTML 清洗本身的影响

本次 `full` 是"去标签后的完整 query"，与基线原始 query（含 `<p>` 等）相比：
Bitcoin dev 0.2171→0.2152、train 0.1550→0.1647；Economics dev 0.1891→0.1768、
train 0.1209→0.1262；IOTA dev 0.4712→0.4553、train 0.2131→0.2083。
方向不一致、幅度 ±0.01，说明单纯去 HTML 标签不是稳定增益，**不能**把它当作改进。

## 文件

* 脚本：`scripts/run_dense_query_variants.py`
* run：`runs/dense_{full,title,summary,title_summary}_<domain>_<split>.trec`
* 评分：`results/score_dense_<mode>_<domain>_<split>.json`，meta：`results/dense_<mode>_*.meta.json`
