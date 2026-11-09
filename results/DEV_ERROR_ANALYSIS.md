# Dev error analysis: SFT reranker vs V1 RRF (n = 519 queries, 13 domains)

Source: `results/score_sub_sft11_<domain>_dev.json` vs `results/score_sub_1_1_<domain>_dev.json` (per-query nDCG@10, Rec@100, num_gold), query text from `data/track1_tempo/<domain>/examples_dev.jsonl`, step counts from `steps_dev.jsonl` (484/519 queries). No model runs; arithmetic only. Domain-macro nDCG@10: SFT 0.33356, V1 0.31660. "Tie" below means |Δ| < 1e-12.

## 1. Where the damage is

- SFT scores exactly **0.0000 on 144/519 queries (27.7%)**; V1 scores 0 on 151. Of the 144 SFT zeros, **131 are persistent** (V1 also 0) and **13 are SFT-specific regressions** (V1 > 0) — the 13 are the complete set, not a sample.
- By candidate-pool recall of the zero queries: **92/144 (64%) have gold inside the dense top-100** (`Rec@100 > 0`) → reordering failure a reranker could fix; **52/144 (36%) have `Rec@100 = 0`** → retrieval failure no top-100 reranker can fix. Among the 131 persistent zeros: 79 in-pool, 52 retrieval.
- Net effect over all 519 (vs V1): 169 improved (+18.02 nDCG pts), 106 worsened (−10.72), 244 exactly tied; net +7.30, consistent with the +0.017 domain-macro gain.

## 2. The "15 worst" (144-way tie, stated honestly)

All 144 zeros are exactly 0.0000, so the worst 15 is **not identifiable** from nDCG@10 alone. The table is the **13 regressions** (the only strictly ordered subset inside the zeros, sorted by V1) plus **2 persistent zeros chosen to cover the two mechanisms**: one in-pool miss (history/70583_271, `Rec@100=1.0`, gold in pool but neither system retrieves it in top-10) and one retrieval miss (travel/201562_1094, `Rec@100=0`).

| # | domain | qid | SFT | V1 | Rec@100 | gold | steps | years in query | query (trimmed) |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | bitcoin | 112359_73 | 0.0000 | 0.3333 | 1.00 | 1 | – | 2017, 2018, 2020 | Why were some BIPs marked as Rejected before expiry deadline defined in BIP-2? |
| 2 | history | 72564_153 | 0.0000 | 0.3333 | 1.00 | 1 | 2 | – | Has a region/territory/country using the Western 7-day week ever got "out-of-sync" for the day of the week? |
| 3 | monero | 13996_845 | 0.0000 | 0.3155 | 1.00 | 1 | 1 | 2038 | Is Monero affected by the year 2038 / 2106 problem? |
| 4 | cardano | 4225_128 | 0.0000 | 0.2044 | 1.00 | 2 | 3 | – | Is there a way for a validator to get the current time? |
| 5 | politics | 89697_946 | 0.0000 | 0.2044 | 1.00 | 2 | 3 | – | Why has the AP still not called some House races that appear mathematically impossible...? |
| 6 | history | 70693_263 | 0.0000 | 0.1846 | 1.00 | 2 | 2 | – | How did Raphael paint halos? |
| 7 | quant | 83939_1059 | 0.0000 | 0.1772 | 0.50 | 2 | 3 | – | How can we determine the minimum viable holding period given granular data and transaction costs? |
| 8 | history | 70624_268 | 0.0000 | 0.1681 | 0.75 | 4 | 3 | 1969 | What was the shortest time between a historical event occurring and a museum opening dedicated to it? |
| 9 | history | 70131_301 | 0.0000 | 0.1413 | 1.00 | 3 | 1 | 2017, 2022 | What were the historical antecedents of Russia BTG-based military organization? |
| 10 | travel | 183966_1189 | 0.0000 | 0.1413 | 1.00 | 3 | 3 | 2023, 2024 | What is the situation with land border crossings on the Georgia-Azerbaijan border? |
| 11 | history | 69226_361 | 0.0000 | 0.1357 | 1.00 | 3 | 1 | – | Why did water stored on premodern ships "go brackish" or "go bad" while at sea? |
| 12 | history | 63331_580 | 0.0000 | 0.1301 | 0.50 | 4 | 3 | – | Did a unit of 40 men hold off Rommel's "ghost division" for 18 days? |
| 13 | history | 64604_517 | 0.0000 | 0.0979 | 0.71 | 7 | 3 | 2008 | Was the financial crash of 2008 caused by top graduates of the Ivy Leagues working at wall street banks? |
| 14 | history | 70583_271 | 0.0000 | 0.0000 | 1.00 | 1 | 2 | – | How do we know when Nero was born? |
| 15 | travel | 201562_1094 | 0.0000 | 0.0000 | 0.00 | 2 | 1 | – | What's the limit, if any, on the number of stops or duration that a single PNR may contain? |

## 3. Regressions vs persistent failures

- All 13 regressions have gold in the pool (12 at `Rec@100 ≥ 0.5`, 9 at 1.0), so they are **pure ordering damage inside the frozen candidate set**: V1 put a relevant doc in the top-10, SFT pushed every one below it. They are few-gold queries (gold 1–7, median 2), where one or two misplaced items zero the score.
- The 13 carry 2.57 of the 10.72 nDCG points lost to V1 (24%); the other 76% is spread over 93 smaller drops (mean loss 0.088/query vs mean gain 0.107 over the 169 wins).
- Year character of the 13: **3 have ≥2 explicit years**, 3 a single year, **7 no explicit year at all** (implicit period, e.g. "premodern ships", "why did Raphael paint halos"). The 3/13 = 23% multi-year share is not above the 19.4% base rate of the dev set (Wilson 95% CI for 3/13: 6–52%), so "year-mismatch queries regress" is **not supported** at this n.
- Persistent failures are the bigger bucket (131): 60% in-pool, 40% retrieval — headroom no reranker can reach.

## 4. Query-character comparison (null result)

| feature | zeros (n=144) | non-zeros (n=375) | difference |
|---|---:|---:|---|
| mentions any explicit year | 34.0% (49) | 35.7% (134) | −1.7pp, 95% CI [−10.9, +7.5] |
| mentions ≥2 distinct years | 19.4% (28) | 18.7% (70) | +0.8pp, 95% CI [−6.7, +8.3] |
| mean official steps (steps subset) | 2.33 (n=120) | 2.57 (n=364) | −0.24, sign flips within domains |
| median gold docs / median query tokens | 3 / 128 | 3 / 139 | ~8% shorter queries |

- The aggregate −0.24 step difference is a **domain-composition artifact**: within domains the sign flips (bitcoin 2.09 vs 2.06, politics 3.00 vs 2.58, history 2.64 vs 2.76), and per-domain zero rates range from 0% (iota, n=3) to 50% (quant, n=10), 47% (bitcoin, n=30).
- No query-visible feature we can compute separates failures; the only strong predictor is **whether the dense top-100 contains the gold doc**.

## 5. Uncertainty and caveats

- n=13 regressions is small: any rate carries ±20–30pp; treat §3's year breakdown as anecdote, not a finding.
- Regressions are only *enumerated for zero-scoring queries*; SFT may also damage queries that stay positive (net effect is +7.30, so partial damage is real but not isolated here).
- `TC@10`/`TP@10` are not independent evidence: step gold ids are a subset of the qrels gold (1850/1850), so `TC@10 = 0` whenever `nDCG@10 = 0`.
- Single run per system; no seed variance is available, so 13-vs-131 is a decomposition, not a significance test.
