p = r'D:/RETECO/docs/RESEARCH_LOG.md'
s = open(p, encoding='utf-8').read()

anchor = '## 6. Agent 协作记录'
add = '''## 5b. Phase 3：基线复现根因 + 时间特征消融（已完成）

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

'''
assert anchor in s
s = s.replace(anchor, add + anchor)
open(p, 'w', encoding='utf-8').write(s)
print('log updated')
