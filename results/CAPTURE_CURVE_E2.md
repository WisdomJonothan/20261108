# Capture curve: how much of the frozen-top-100 oracle gap the submitted system actually captures

## 2026-11-09 - capture curve (c92c9e08dcd54818)

Runs: `runs/dense_mxbai_{domain}_dev.trec` vs `runs/sub_sft_e2bc_{domain}_dev.trec`, dev split, k=10, pool depth=100.

| domain | n | dense nDCG@10 | oracle nDCG@10 (top-100) | system nDCG@10 | delta system-dense | capture |
|---|---:|---:|---:|---:|---:|---:|
| bitcoin | 30 | 0.1820 | 0.6349 | 0.2143 | +0.0322 | 7.1% |
| cardano | 15 | 0.1832 | 0.7819 | 0.2258 | +0.0426 | 7.1% |
| economics | 25 | 0.1890 | 0.6769 | 0.2512 | +0.0622 | 12.7% |
| genealogy | 35 | 0.3298 | 0.6334 | 0.4060 | +0.0761 | 25.1% |
| history | 240 | 0.3051 | 0.8277 | 0.3918 | +0.0867 | 16.6% |
| hsm | 45 | 0.3125 | 0.7937 | 0.4651 | +0.1526 | 31.7% |
| iota | 3 | 0.6485 | 0.9440 | 0.8383 | +0.1898 | 64.2% |
| law | 11 | 0.3689 | 0.8720 | 0.4357 | +0.0667 | 13.3% |
| monero | 19 | 0.1826 | 0.6655 | 0.2069 | +0.0242 | 5.0% |
| politics | 45 | 0.3599 | 0.7017 | 0.4290 | +0.0690 | 20.2% |
| quant | 10 | 0.1954 | 0.5559 | 0.1538 | -0.0416 | -11.6% |
| travel | 30 | 0.2321 | 0.7787 | 0.2494 | +0.0173 | 3.2% |
| workplace | 11 | 0.3362 | 0.8138 | 0.3881 | +0.0518 | 10.9% |
| **domain-macro (official)** | 519 | 0.2943 | 0.7446 | 0.3581 | +0.0638 | **14.2%** |

- baseline = dense mxbai first stage; oracle = ideal ordering of the frozen top-100; capture = (system - baseline) / (oracle - baseline), computed on the three domain-macro means.
- unweighted mean of the 13 per-domain capture ratios = 15.8% (gives a 3-query domain the same weight as 240-query history).
- flat topic-macro capture over all 519 queries = 15.7%.
- depth sensitivity: capture against the 1000-deep oracle (not reachable by a top-100 reranker) = 9.9%.
- per-query capture: mean 15.3%, median 8.8%; 47.4% of queries improve, 16.4% get worse.
<!-- capture-curve:c92c9e08dcd54818 -->
