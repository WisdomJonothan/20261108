# 为什么"官方预建 Lucene 索引"复现不了官方发布的 BM25 基线

结论先行：**我们复现了官方基线，而且逐位一致（0.3309/0.1563/0.0370/0.0856）；
复现不了的是"用 Lucene BM25 复现官方基线"这件事** —— 因为官方基线用的并不是
Lucene 的 BM25 打分函数，而是 gensim 的 `LuceneBM25Model`，名字里带 Lucene，
公式并不等价。

## 1. 事实

| 系统 | Bitcoin dev nDCG@10 | IOTA dev | 说明 |
| --- | --- | --- | --- |
| 官方 starter-kit（gensim `LuceneBM25Model`） | **0.0370** | **0.3309** | 与发布值逐位一致 |
| 官方预建 Lucene 索引（Anserini） | 0.0300 | 0.3852 | 不同 |
| 我们自建 Anserini 索引（同参数、同 jar） | **0.0300** | — | 与预建索引完全一致 |

→ 预建索引没有问题：**用 Anserini 重新建一遍索引，分数和官方预建索引一模一样**。
差异不在索引，在打分函数。

## 2. 两个打分函数

Lucene 8/9 的 BM25（`BM25Similarity`，常数 (k1+1) 被省略，不影响排序）：

```
score(q,d) = Σ_t  idf(t) * f(t,d) / (f(t,d) + k1 * (1 - b + b * dl(d)/avgdl))
idf(t)     = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
```

gensim `LuceneBM25Model` 被官方脚本这样用：先 `model[corpus]` 得到文档向量，
再 `model[doc2bow(query)]` 得到查询向量，然后做**点积**：

```
doc_vec_t  = idf(t) * f(t,d) / (f + k1*(1 - b + b*dl/avgdl))
query_vec_t= idf(t) * f(t,q) / (f(t,q) + k1*(1 - b + b*dl(q)/avgdl))
score      = Σ_t doc_vec_t * query_vec_t          ← idf 出现两次
```

即：**官方基线是"idf 平方 + 查询长度归一化"的 BM25。** 两个实现的 idf 公式其实
相同（`log((N+1)/(df+0.5))` 与 Lucene 的写法恒等），analyzer 也相同（
`io.anserini.analysis.DefaultEnglishAnalyzer`，分词逐 token 一致）。

## 3. 数值验证（同一 query、同一文档，IOTA dev `2714_798` / `c5cc10bf_0004`）

| 量 | 数值 |
| --- | --- |
| 官方流水线实际得分 | 93.5363 |
| 用 gensim 公式手算 | 93.4127（差 0.13%，来自字典统计细节） |
| Lucene 索引实际得分 | 77.2161 |
| 用索引自身统计（docCount/df/dlv）手算 Lucene 公式 | 77.1619（差 0.07%，来自 norm 量化） |
| Anserini `compute_query_document_score` | 77.2161 ✓ |

查询内 68 个不同词的 gensim 权重跨度 0.394–5.318（13.5 倍），所以 idf 平方
会显著改变词项相对权重——这就是同一 query 下两套系统 top-10 只有约 60% 重合、
nDCG 从 0.0370 变到 0.0300 的原因。

## 4. 其他已排除的解释

* **analyzer 不同**：排除。两套系统的分词输出逐 token 一致（含停用词处理）。
* **k1/b 不同**：排除。双方都是 0.9/0.4。
* **索引不同**：排除。自建索引与预建索引分数完全一致（0.0300）。
* **文档长度归一化量化（norm 量化到 1 字节）**：只造成 ~0.07% 偏差，不足以解释差距。
* **(k1+1) 因子**：Lucene 省略它（常数，不影响排序）；gensim 也省略。不是原因。

## 5. 结论与含义

1. **发布基线 = gensim `LuceneBM25Model` 的"idf²"BM25**，任何标准 Lucene/Anserini
   BM25 都复现不了它；反过来也一样。
2. 论文里必须区分两者：说"official BM25 baseline"时指的是组织者代码，不是 Lucene。
3. 我们后续实验统一用官方代码作为"官方基线"，用 Anserini 索引作为"可扩展 BM25"，
   两者不混比。
4. 复现性建议：若要第三方复现，必须锁定 `gensim` 版本（本机 4.3.3）与
   `LuceneBM25Model` 的实现，而不仅是 k1/b。

## 6. 未决的小问题（不影响结论）

* 预建索引的 norm 量化会引入 ~0.07% 的单点分数偏差；批量 nDCG 上未单独量化。
* gensim 手算与官方实现有 0.13% 的差距，来自 `Dictionary` 的 df/文档长度统计细节
  （例如空文档、重复文档处理），未逐项追查。
