p = r'D:/RETECO/docs/RESEARCH_LOG.md'
s = open(p, encoding='utf-8').read()

s = s.replace(
    '| Q3 | 零样本 dense 与 BM25 谁强 | dense 显著更强（Bitcoin dev 0.2171 vs 0.0300） | Phase 0 §4 |',
    '| Q3 | 零样本 dense 与 BM25 谁强 | dense 显著更强（Bitcoin dev：dense 0.2171 vs 官方复现 BM25 0.0370；本地预建索引 BM25 0.0300） | Phase 0 §4 |')

s = s.replace(
    '* 成本：45 次 LLM 调用/query ≈ 11 秒/query，约为 dense 检索（4.5 ms）的 2400 倍；',
    '* 成本：45 次 LLM 调用/query，Bitcoin dev 353 s/1350 次 = **11.8 s/query**（Economics 11.2 s/query），约为 dense 检索（4.5 ms）的 2600 倍；\n'
    '* listwise 的"复读"已用三项证据确认：原始输出就是 `[1] > [2] > ...`；解析失败 **0/30**；**把候选顺序整体反转后仍输出 `[1]..[20]`**（即输出的是输入位置而非相关性）；')

s = s.replace(
    '* 结论：小模型有 pairwise 判别力（探针 71.7% 准确率、位置偏差 51.7%），但不能做 listwise；listwise 要成立需要更大模型或 API，成本仍是问题。',
    '* 结论：小模型有 pairwise 判别力（探针 71.7% 准确率、位置偏差 51.7%），但在该 checkpoint+prompt 下不能做 listwise；要验证 listwise 需要更大模型或 API，成本仍是问题。\n'
    '* 注意：pairwise 是 depth-10 实验，pointwise 是 depth-100 实验，两者不可当作同一比较；与 dense 的比较各自都在相同候选集上。')

open(p, 'w', encoding='utf-8').write(s)
print('patched log')
