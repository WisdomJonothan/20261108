# RETECO Track 1a — Phase 0 report (IOTA / Bitcoin)

All numbers below were produced on this machine from the official v1.1 release and the official Pyserini prebuilt indexes; none are quoted from the organizers except the `published` column used for verification.

## Environment

- Python: `3.12.7` (D:\anaconda\python.exe)
- Java: `openjdk version "21.0.13" 2026-10-20 LTS`
- Packages: pyserini 2.4.0, pyjnius 1.9.0, gensim 4.3.3, numpy 1.26.4, scipy 1.13.1, pandas 2.2.2, torch 2.7.0+cu126, transformers 5.22.2, huggingface-hub 2.4.0, tqdm 4.66.5
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (torch CUDA 12.6)

## BM25: official starter-kit baseline vs official prebuilt Lucene index

| domain | split | topics | official nDCG@10 | published | prebuilt index nDCG@10 | R@100 off | R@100 idx | oracle@10 idx (cand100) |
|---|---|---|---|---|---|---|---|---|
| iota | dev | 3 | 0.3309 | 0.3309 | 0.3852 | 0.9167 | 0.5000 | 0.5456 |
| iota | train | 7 | 0.1563 | 0.1563 | 0.0352 | 0.3619 | 0.2714 | 0.3464 |
| bitcoin | dev | 30 | 0.0370 | 0.0370 | 0.0300 | 0.2820 | 0.2848 | 0.3314 |
| bitcoin | train | 70 | 0.0856 | 0.0856 | 0.0468 | 0.2833 | 0.1488 | 0.1833 |

## Dense (zero-shot bge-base-en-v1.5) vs BM25 vs candidate union

| domain | split | BM25 nDCG@10 | Dense nDCG@10 | Dense R@100 | Union R@100 | BM25 R@100 | oracle@10 union (cand100) |
|---|---|---|---|---|---|---|---|
| iota | dev | 0.3852 | 0.4712 | 1.0000 | 1.0000 | 0.5000 | 1.0000 |
| iota | train | 0.0352 | 0.2131 | 0.4071 | 0.5000 | 0.2714 | 0.6294 |
| bitcoin | dev | 0.0300 | 0.2171 | 0.5688 | 0.5688 | 0.2848 | 0.6488 |
| bitcoin | train | 0.0468 | 0.1550 | 0.5769 | 0.5890 | 0.1488 | 0.6532 |

## Candidate-depth sweep (recall and candidate-oracle at the same depth)

`oracle` = ideal re-ranking of exactly the top-k candidates, so `oracle == 0` whenever `recall == 0` (asserted for every query). `union` uses a 2k budget (diagnostic); `RRF@k` is the equal-budget fusion, fused from the full lists and truncated to k, scored for real.

| domain | split | k | BM25 rec | BM25 orc | dense rec | dense orc | union rec | union orc | RRF nDCG@10 | RRF rec |
|---|---|---|---|---|---|---|---|---|---|---|
| iota | dev | 20 | 0.3056 | 0.3852 | 0.6667 | 0.7499 | 0.6667 | 0.7499 | 0.4673 | 0.6667 |
| iota | dev | 50 | 0.3889 | 0.4673 | 0.8333 | 0.8710 | 0.8333 | 0.8710 | 0.4916 | 0.8333 |
| iota | dev | 100 | 0.5000 | 0.5456 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.4876 | 1.0000 |
| iota | dev | 200 | 0.9167 | 0.9440 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5797 | 1.0000 |
| iota | dev | 1000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5797 | 1.0000 |
| iota | train | 20 | 0.1667 | 0.2250 | 0.2310 | 0.3238 | 0.3143 | 0.4261 | 0.2118 | 0.2024 |
| iota | train | 50 | 0.1667 | 0.2250 | 0.3595 | 0.4724 | 0.4429 | 0.5746 | 0.2012 | 0.3143 |
| iota | train | 100 | 0.2714 | 0.3464 | 0.4071 | 0.5395 | 0.5000 | 0.6294 | 0.2254 | 0.4429 |
| iota | train | 200 | 0.4000 | 0.5017 | 0.4905 | 0.6082 | 0.6000 | 0.7054 | 0.2647 | 0.5000 |
| iota | train | 1000 | 0.7000 | 0.7851 | 0.7952 | 0.8519 | 0.8667 | 0.9040 | 0.2957 | 0.7000 |
| bitcoin | dev | 20 | 0.1233 | 0.1403 | 0.3849 | 0.4521 | 0.3932 | 0.4603 | 0.1551 | 0.3551 |
| bitcoin | dev | 50 | 0.2119 | 0.2382 | 0.4794 | 0.5550 | 0.4794 | 0.5550 | 0.1514 | 0.4124 |
| bitcoin | dev | 100 | 0.2848 | 0.3314 | 0.5688 | 0.6488 | 0.5688 | 0.6488 | 0.1513 | 0.5110 |
| bitcoin | dev | 200 | 0.3542 | 0.3931 | 0.6213 | 0.6965 | 0.6463 | 0.7150 | 0.1595 | 0.5966 |
| bitcoin | dev | 1000 | 0.4907 | 0.5429 | 0.8450 | 0.8883 | 0.8700 | 0.9068 | 0.1647 | 0.8206 |
| bitcoin | train | 20 | 0.0976 | 0.1144 | 0.3045 | 0.3603 | 0.3367 | 0.3945 | 0.1183 | 0.2317 |
| bitcoin | train | 50 | 0.1248 | 0.1514 | 0.4595 | 0.5218 | 0.4802 | 0.5465 | 0.1154 | 0.3660 |
| bitcoin | train | 100 | 0.1488 | 0.1833 | 0.5769 | 0.6385 | 0.5890 | 0.6532 | 0.1146 | 0.4879 |
| bitcoin | train | 200 | 0.2683 | 0.3138 | 0.6627 | 0.7181 | 0.6775 | 0.7333 | 0.1178 | 0.6055 |
| bitcoin | train | 1000 | 0.4914 | 0.5334 | 0.8287 | 0.8613 | 0.8394 | 0.8693 | 0.1198 | 0.8015 |

## Temporal feasibility (content-time signals in the released corpus)

| domain | docs | with 4-digit year | ISO date | month name | temporal-flagged queries | gold docs with a year |
|---|---|---|---|---|---|---|
| iota | 9680 | 21.2% | 0.5% | 25.2% | 3/3 | 25.0% |
| bitcoin | 76706 | 23.9% | 3.8% | 28.4% | 26/30 | 12.1% |

## Per-query comparison: dense vs BM25 (same candidate budget, top-1000)

| domain | split | queries | dense wins | ties | BM25 wins | BM25 nDCG@10=0 | dense nDCG@10=0 | BM25 no gold@100 | dense no gold@100 |
|---|---|---|---|---|---|---|---|---|---|
| iota | dev | 3 | 2 | 0 | 1 | 1 | 0 | 1 | 0 |
| iota | train | 7 | 4 | 3 | 0 | 6 | 3 | 3 | 0 |
| bitcoin | dev | 30 | 17 | 12 | 1 | 27 | 12 | 15 | 3 |
| bitcoin | train | 70 | 28 | 39 | 3 | 60 | 40 | 48 | 11 |

## Cross-check: official `scorer.py` on the same run files

| run | our nDCG@10 | official scorer nDCG@10 | |diff| | TP diff | TC diff |
|---|---|---|---|---|---|
| bm25_official iota dev | 0.3309 | 0.3309 | 0.0000 | 0.0000 | 0.0000 |
| bm25_pyserini iota dev | 0.3852 | 0.3852 | 0.0000 | 0.0000 | 0.0000 |
| dense_bge iota dev | 0.4712 | 0.4712 | 0.0000 | 0.0000 | 0.0000 |
| bm25_official iota train | 0.1563 | 0.1563 | 0.0000 | 0.0000 | 0.0000 |
| bm25_pyserini iota train | 0.0352 | 0.0352 | 0.0000 | 0.0000 | 0.0000 |
| dense_bge iota train | 0.2131 | 0.2131 | 0.0000 | 0.0000 | 0.0000 |
| bm25_official bitcoin dev | 0.0370 | 0.0370 | 0.0000 | 0.0000 | 0.0000 |
| bm25_pyserini bitcoin dev | 0.0300 | 0.0300 | 0.0000 | 0.0000 | 0.0000 |
| dense_bge bitcoin dev | 0.2171 | 0.2171 | 0.0000 | 0.0000 | 0.0000 |
| bm25_official bitcoin train | 0.0856 | 0.0856 | 0.0000 | 0.0000 | 0.0000 |
| bm25_pyserini bitcoin train | 0.0468 | 0.0468 | 0.0000 | 0.0000 | 0.0000 |
| dense_bge bitcoin train | 0.1550 | 0.1550 | 0.0000 | 0.0000 | 0.0000 |

Maximum absolute difference across all 12 runs: **0.000047** (the scoring path used for the tables above is therefore the official metric).

## Retrieval latency (warm, IOTA index, 3 repetitions per query)

- top-10: mean 46.3 ms, p50 25.6 ms
- top-100: mean 95.1 ms, p50 67.3 ms
- top-1000: mean 265.5 ms, p50 260.7 ms
