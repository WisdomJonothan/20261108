# Capture curve: how much of the frozen-top-100 oracle gap the submitted system actually captures

## 2026-11-09 - capture curve (f3df9535001fe155)

Runs: `runs/dense_mxbai_{domain}_dev.trec` vs `runs/sub_sft11_{domain}_dev.trec`, dev split, k=10, pool depth=100.

| domain | n | dense nDCG@10 | oracle nDCG@10 (top-100) | system nDCG@10 | delta system-dense | capture |
|---|---:|---:|---:|---:|---:|---:|
| bitcoin | 30 | 0.1820 | 0.6349 | 0.2085 | +0.0265 | 5.8% |
| cardano | 15 | 0.1832 | 0.7819 | 0.2047 | +0.0215 | 3.6% |
| economics | 25 | 0.1890 | 0.6769 | 0.2157 | +0.0267 | 5.5% |
| genealogy | 35 | 0.3298 | 0.6334 | 0.3838 | +0.0539 | 17.8% |
| history | 240 | 0.3051 | 0.8277 | 0.3547 | +0.0496 | 9.5% |
| hsm | 45 | 0.3125 | 0.7937 | 0.4120 | +0.0995 | 20.7% |
| iota | 3 | 0.6485 | 0.9440 | 0.8344 | +0.1859 | 62.9% |
| law | 11 | 0.3689 | 0.8720 | 0.3691 | +0.0001 | 0.0% |
| monero | 19 | 0.1826 | 0.6655 | 0.2048 | +0.0222 | 4.6% |
| politics | 45 | 0.3599 | 0.7017 | 0.4074 | +0.0475 | 13.9% |
| quant | 10 | 0.1954 | 0.5559 | 0.1547 | -0.0407 | -11.3% |
| travel | 30 | 0.2321 | 0.7787 | 0.2475 | +0.0154 | 2.8% |
| workplace | 11 | 0.3362 | 0.8138 | 0.3391 | +0.0028 | 0.6% |
| **domain-macro (official)** | 519 | 0.2943 | 0.7446 | 0.3336 | +0.0393 | **8.7%** |

- baseline = dense mxbai first stage; oracle = ideal ordering of the frozen top-100; capture = (system - baseline) / (oracle - baseline), computed on the three domain-macro means.
- unweighted mean of the 13 per-domain capture ratios = 10.5% (gives a 3-query domain the same weight as 240-query history).
- flat topic-macro capture over all 519 queries = 9.4%.
- depth sensitivity: capture against the 1000-deep oracle (not reachable by a top-100 reranker) = 6.1%.
- per-query capture: mean 8.0%, median 0.0%; 41.4% of queries improve, 19.1% get worse.
<!-- capture-curve:f3df9535001fe155 -->
