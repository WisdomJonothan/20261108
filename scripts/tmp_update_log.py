import io

p = r'D:/RETECO/docs/RESEARCH_LOG.md'
s = open(p, encoding='utf-8').read()
old = '**结果**：见 `results/PHASE2_REPORT.md`（本文件在结果落盘后补充数字）。'
new = '''**结果**

| 系统 | Economics dev (25) | Economics train (58) |
|---|---|---|
| BM25 (Lucene) | 0.0269 | 0.0361 |
| BGE dense | **0.1891** | **0.1209** |
| pointwise v2-m3（标题式） | 0.1664 | — |
| oracle@100 | 0.6460 | 0.6008 |

LLM 精排（Qwen2.5-1.5B-Instruct）：

* **listwise（window 20）在 IOTA/Bitcoin 上输出 = 输入顺序**（0.4712 / 0.2171，与 dense 完全相同）；原始输出就是 `[1] > [2] > ...`，属于小模型的 primacy 复读，是方法失败而非结论；
* **pairwise（top-10 全比较）**：Bitcoin dev 0.2171 → **0.2602**（+0.0431，CI [−0.021,+0.116]）；Economics dev 0.1891 → **0.2082**（+0.0191，CI [−0.012,+0.055]）；合并 55 条 **+0.0322（CI [−0.007,+0.077]，P=0.945）**；与 dense 顺序的位置一致率仅 12.7%，确有重排；
* 成本：45 次 LLM 调用/query ≈ 11 秒/query，约为 dense 检索（4.5 ms）的 2400 倍；
* 结论：小模型有 pairwise 判别力（探针 71.7% 准确率、位置偏差 51.7%），但不能做 listwise；listwise 要成立需要更大模型或 API，成本仍是问题。'''
assert old in s, 'anchor not found'
s = s.replace(old, new)
s = s.replace('## 5. Phase 2：跨领域验证 + listwise 精排（进行中）',
              '## 5. Phase 2：跨领域验证 + listwise/pairwise 精排（已完成）')
open(p, 'w', encoding='utf-8').write(s)
print('log updated')
