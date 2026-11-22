import io, json
from pathlib import Path
R = Path("results/replacement_v2")
res = {"bitcoin": [0.2199, 0.2143], "quant": [0.1872, 0.1538], "law": [0.4061, 0.4357]}
ok = Path("results/replacement_v2/embedder_swap_probe_dev.json")
ok.write_text(json.dumps({
    "probe_domains": ["bitcoin", "quant", "law"],
    "dense_qwen3emb06_ndcg10": {"bitcoin": 0.1824, "quant": 0.1698, "law": 0.3860},
    "dense_mxbai_ndcg10": {"bitcoin": 0.1820, "quant": 0.1954, "law": 0.3689},
    "dense_qwen3emb06_recall100": {"bitcoin": 0.5794, "quant": 0.5167, "law": 0.9409},
    "dense_mxbai_recall100": {"bitcoin": 0.5646, "quant": 0.4833, "law": 0.8545},
    "fused_qwen3pool": {k: v[0] for k, v in res.items()},
    "fused_frozen": {k: v[1] for k, v in res.items()},
    "mean_fused_delta_three_domains": round(sum(v[0] - v[1] for v in res.values()) / 3, 6),
    "note": "Same frozen CE checkpoint and z 1:1 fusion; only the first-stage dense run changed. "
            "Three of thirteen dev domains, so the implied full-macro effect is one thirteenth of "
            "the per-domain mean times three.",
}, indent=2), encoding="utf-8")

rep = R / "REPORT.md"
s = rep.read_text(encoding="utf-8")
block = """
## 12. Swapping the first-stage embedder (three-domain probe)

Qwen3-Embedding-0.6B replaced mxbai-embed-large-v1 on three dev domains (bitcoin, quant, law),
everything downstream unchanged (same frozen CE checkpoint, same z 1:1 fusion, same detie).

| domain | dense nDCG@10 mxbai -> qwen3 | R@100 mxbai -> qwen3 | fused mxbai -> qwen3 | delta |
| --- | --- | --- | --- | --- |
| bitcoin | 0.1820 -> 0.1824 | 0.5646 -> 0.5794 | 0.2143 -> 0.2199 | +0.0056 |
| quant | 0.1954 -> 0.1698 | 0.4833 -> 0.5167 | 0.1538 -> 0.1872 | +0.0334 |
| law | 0.3689 -> 0.3860 | 0.8545 -> 0.9409 | 0.4357 -> 0.4061 | -0.0295 |

Recall@100 improves in all three domains (+1.5 to +8.6 points) but the fused score is mixed:
quant gains a third of its score, law loses three points, and the mean over the three domains is
+0.0032, i.e. roughly +0.0007 on the 13-domain macro. Encoding the remaining ten domains takes
about eight hours on this contended machine, so the expected return does not justify it under the
current evidence; the probe is recorded as a boundary result (better coverage does not translate
into a better slate when the cross-encoder is frozen).
"""
if "Swapping the first-stage embedder" not in s:
    s = s.replace("## 11. Score-side experiments", block + "\n## 11. Score-side experiments", 1) if "## 11. Score-side experiments" in s else s.rstrip() + block
    rep.write_text(s, encoding="utf-8")

one = Path("results/replacement_v2/one.json")
print("ok")
