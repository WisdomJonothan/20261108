# 时间约束的表示：查询分解（Phase 7B）

## 问题

前四轮的时间实验（字面年份、内容时间特征、语义时间陈述、query 改写）都没有稳定收益。
本轮换一种"时间约束表示"：**把查询分解成若干时间子问题**，分别检索再融合。这是
temporal IR 里最常见的做法，也对应 Track 1b 的 step 结构。

两个版本：

* **oracle decomposition**：直接用官方的 `steps_*.jsonl`（每条 query 的
  `step_instruction`）作为子查询——这代表"分解是完美的"时的上限（注意：steps 是为
  1b 设计的，不保证与 1a 的 gold 对齐）；
* **LLM decomposition**：用本地 Qwen2.5-1.5B 把 query 拆成 2–4 个时间子问题——这是
  测试时真正可得的版本。

融合：每个子查询各取 top-1000，按 RRF 累加；原 query 的权重为 `w0`（w0 越大越接近
单 query 检索，w0→∞ 即基线）。

## 结果（Bitcoin dev / Economics dev，dense 基线 0.2171 / 0.1891）

### Oracle decomposition（官方 steps）

| w0 | Bitcoin dev | Economics dev |
|---|---|---|
| 1 | 0.1902 | 0.1383 |
| 2 | 0.2196 | 0.1473 |
| 4 | **0.2393** | 0.1621 |
| 8 | 0.2333 | 0.1652 |

### LLM decomposition（Qwen2.5-1.5B）

| w0 | Bitcoin dev | Economics dev |
|---|---|---|
| 1 | 0.1664 | 0.1186 |
| 2 | 0.1812 | 0.1520 |
| 4 | 0.2012 | 0.1613 |

## 结论

1. **LLM 分解在所有配置下都低于基线**（Bitcoin 最好 0.2012 vs 0.2171；Economics
   0.1613 vs 0.1891）。1.5B 模型生成的子问题质量不足以支撑分解式检索。
2. **即使给"完美"的官方步骤，也只有 Bitcoin 在 w0=4 时略高于基线**（0.2393 vs 0.2171），
   Economics 始终更低（最好 0.1652 vs 0.1891）。而且官方 steps 面向 1b、与 1a 的 gold
   并不对齐，这个 0.2393 不能当成真正的上限。
3. 与前几轮一致：**时间约束的价值不在"多问几个子问题"**，而在于能否获得独立的时间证据
   （语料里没有发布时间，gold 只有 12% 含显式年份）。

## 文件

* `scripts/run_decomposed_retrieval.py`、`scripts/decompose_temporal.py`
* `runs/decomp_steps_*`、`runs/decomp_llm_*`、`results/decomp_llm_*.json`（含原始生成）
* 评分：`results/score_decomp_*.json`
