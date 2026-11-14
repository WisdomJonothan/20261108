# Qwen3-Reranker-0.6B baseline (dev)

Protocol: official model card (`prefix + budgeted body + think suffix`, P(yes) from `[no, yes]`), `max_len=2048`, `doc_cap=4000` chars, batch 4-16.
Candidates: the same dense top-100 per query as our reranker.

| domain | Qwen CE alone | Qwen + dense (1:1) | our system (frozen recipe) |
|---|---|---|---|
| bitcoin | 0.1534 | 0.1651 | 0.2144 |
| cardano | 0.1401 | 0.1807 | 0.2270 |
| economics | 0.1277 | 0.1940 | 0.2366 |
| genealogy | 0.2493 | 0.3455 | 0.4074 |
| history | 0.2503 | 0.3039 | 0.3997 |
| hsm | 0.2577 | 0.3217 | 0.4671 |
| iota | 0.5495 | 0.7249 | 0.8411 |
| law | 0.3004 | 0.3512 | 0.4274 |
| monero | 0.1664 | 0.2316 | 0.2052 |
| politics | 0.3114 | 0.3544 | 0.4326 |
| quant | 0.1412 | 0.1954 | 0.1538 |
| travel | 0.2240 | 0.2620 | 0.2482 |
| workplace | 0.1361 | 0.2648 | 0.3873 |
| **macro** | **0.2313** | **0.2996** | **0.3575** |

Our cross-encoder alone (same probe, same candidates): **0.2765**.
