# Phase 8 — stronger retrievers: full 13-domain dev results

Two runnable encoders were evaluated on the same data (v1.1 Track-1a, dev split),
and the two-view pipeline was rebuilt with the better one. All numbers are dev
nDCG@10 computed by `scripts/score_run.py` (official `ir_metrics`), and the
fusion tables are recomputed from the TREC runs by `scripts/analyze_mxbai.py`.

## Encoders

| model | params | dim | status |
|---|---:|---:|---|
| intfloat/e5-base-v2 | 110M | 768 | all 13 domains |
| mixedbread-ai/mxbai-embed-large-v1 | 335M | 1024 | all 13 domains |
| Alibaba-NLP/gte-large-en-v1.5 | 434M | 1024 | **not runnable**: custom RoPE/pack_qkv code raises `AttributeError: 'NewModel' object has no attribute 'get_extended_attention_mask'` with transformers 5.22 |

Pools/prefixes follow each model card: e5 `query: `/`passage: ` mean pooling;
mxbai query instruction "Represent this sentence for searching relevant passages: "
with mean pooling; fp16, max_length 512, batch 64.

## Dense-stage comparison (dev nDCG@10)

| domain | bge-base | e5-base | mxbai |
|---|---:|---:|---:|
| bitcoin | 0.2171 | 0.1224 | 0.1820 |
| economics | 0.1891 | 0.0525 | 0.1890 |
| politics | 0.3074 | 0.2264 | **0.3599** |
| genealogy | 0.2110 | 0.1436 | **0.3298** |
| hsm | 0.2926 | 0.1413 | **0.3125** |
| history | 0.2772 | 0.2413 | **0.3051** |
| law | 0.3013 | 0.0633 | **0.3689** |
| workplace | **0.3795** | 0.1046 | 0.3362 |
| cardano | 0.1496 | 0.1312 | **0.1832** |
| monero | 0.1328 | 0.1139 | **0.1826** |
| quant | 0.0564 | 0.0000 | **0.1954** |
| travel | 0.2466 | 0.1653 | 0.2321 |
| iota (3 q) | 0.4712 | 0.4183 | **0.6485** |
| **macro (13 domains)** | 0.2486 | 0.1480 | **0.2942** |

mxbai wins 10 of 13 domains (loses bitcoin, workplace, travel; economics is a
tie). e5-base is consistently worse — the encoding path was verified against bge
(the same scorer reproduces bge 0.3013 on law exactly), so this is the model, not
a bug; its mean pooling is simply weak on these long HTML documents.

## Two-view pipeline with mxbai as first stage

Candidates: mxbai dense top-100 → bge-reranker-v2-m3 (q_mode=title, q_cap 256,
window 250, max pooling) → RRF over the same candidate set.

| domain | dense | CE | 2:1 | 1:1 | 1:2 | 1:4 |
|---|---:|---:|---:|---:|---:|---:|
| bitcoin | 0.1820 | 0.2251 | 0.2026 | 0.2044 | 0.2247 | 0.2269 |
| economics | 0.1890 | 0.1642 | 0.2157 | 0.2099 | 0.2151 | 0.1987 |
| politics | 0.3599 | 0.4134 | 0.3941 | 0.4023 | 0.4184 | 0.4240 |
| genealogy | 0.3298 | 0.2917 | 0.3272 | 0.3372 | 0.3383 | 0.3327 |
| hsm | 0.3125 | 0.4289 | 0.3641 | 0.3840 | 0.4039 | 0.4334 |
| history | 0.3051 | 0.3188 | 0.3392 | 0.3441 | 0.3459 | 0.3395 |
| law | 0.3689 | 0.2304 | 0.3689 | 0.3574 | 0.2714 | 0.2424 |
| workplace | 0.3362 | 0.2802 | 0.3553 | 0.3244 | 0.3231 | 0.2848 |
| cardano | 0.1832 | 0.1650 | 0.1443 | 0.1632 | 0.1628 | 0.1608 |
| monero | 0.1826 | 0.1517 | 0.2038 | 0.2281 | 0.2143 | 0.1983 |
| quant | 0.1954 | 0.0328 | 0.1535 | 0.1631 | 0.1244 | 0.0640 |
| travel | 0.2321 | 0.2237 | 0.2561 | 0.2205 | 0.2228 | 0.2388 |
| iota (3 q) | 0.6485 | 0.5575 | 0.8036 | 0.7771 | 0.8085 | 0.7309 |

Macro over the 13 domains: dense **0.2943**, CE 0.2680, fused 2:1 **0.3176**,
1:1 0.3166, 1:2 0.3134, 1:4 0.2981.

Paired query bootstrap vs dense (13 domains, 519 queries):

| fusion | Δ | 95% CI | P(Δ>0) |
|---|---:|---|---:|
| 2:1 | +0.0271 | [+0.0163, +0.0377] | 1.000 |
| 1:1 | **+0.0307** | [+0.0166, +0.0449] | 1.000 |
| 1:2 | +0.0334 | [+0.0164, +0.0503] | 1.000 |
| 1:4 | +0.0297 | [+0.0103, +0.0486] | 0.999 |

Domain-clustered bootstrap (13 domains): 2:1 +0.0233 [+0.0009, +0.0492];
1:1 +0.0223 [+0.0012, +0.0471]; 1:2 +0.0191 [−0.0162, +0.0541]; 1:4 +0.0038
[−0.0364, +0.0416]. Leave-one-domain-out weight selection: mean held-out gain
**+0.0087** — small, because per-domain optima differ (law and quant prefer CE,
workplace and iota prefer dense).

## Fusion weight chosen on train (13 domains)

The dev-best weight (2:1) was selected on dev — a reviewer objection. This
experiment chooses the weight using **train only** and reads dev once
(`scripts/fusion_weight_train.py`, `results/fusion_weight_train.json`).

Train macro nDCG@10 over all 13 domains (RRF, dense:CE):

| weight | 1:0 | 4:1 | 2:1 | 1:1 | **1:2** | 1:4 | 0:1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train macro | 0.2350 | 0.2449 | 0.2482 | 0.2491 | **0.2519** | 0.2439 | 0.2271 |

Train picks **1:2** (dense : CE). Applying it unchanged to dev gives a 13-domain
macro of **0.3134** (dense 0.2943; dev-best 2:1 is 0.3176), i.e. **+0.0191 over
dense** with no dev-side tuning. The train curve is very flat around the optimum
(1:1 is 0.0028 lower on train), so the exact weight matters little; what matters
is that CE must carry the larger share.

**Per-domain weights do not transfer.** Choosing a separate weight for each
domain on its own train split and applying it to dev yields a macro of
**0.2941** — indistinguishable from dense, because small train splits pick
degenerate weights (`law` → 1:4, dev 0.2424 vs 0.3689 at 2:1; `hsm` → CE-only).
A single global weight is the only robust choice.

Caveat: with the train-selected weight the domain-clustered CI for the dev gain
is [−0.0162, +0.0541]; the dev-selected 2:1 is the only configuration whose
clustered CI excludes zero. So the honest statement is: the fusion gain is
*positive and stable in point estimate* but not established beyond dev.

## Fair comparison: same 11 domains, both encoders

The old pipeline covered `11 shared + travel`; the new one covers
`11 shared + iota`. IOTA has 3 dev queries and a fused nDCG@10 of 0.7771, so a
12-domain macro that swaps travel for iota is **not comparable** to the old one.
Recomputed on the shared 11 domains / 486 queries
(`scripts/fair_11dom_compare.py`, `results/fair_11dom_compare.json`):

| system | domain macro nDCG@10 |
|---|---:|
| bge-base dense | 0.2285 |
| bge-base + CE, RRF 1:4 | 0.2556 |
| mxbai dense | 0.2677 |
| mxbai + CE, RRF 1:1 | **0.2843** |

* Encoder upgrade: +0.0287 absolute (+12.6 %) over the bge pipeline.
* Fusion gain on mxbai: +0.0166, query bootstrap [+0.0046, +0.0291],
  **domain-clustered [−0.0026, +0.0345] — still crosses zero**.

So: the encoder improvement is solid; the fusion gain is positive in 9 of 11
domains but is not established beyond dev under domain clustering.

## Temporal gate (negative, pre-registered)

A train-fitted temporal gate was evaluated on the separated protocol (fit on
train, λ chosen inside train, dev read once). Result: pooled ΔnDCG@10
**+0.0039** (CI [−0.0023, +0.0103]) — far below the pre-registered +0.02 bar.
See `results/TEMPORAL_GATE.md`. The discriminative signal exists (s_temp AUC
0.61–0.88) but does not convert into ranking gains on these corpora.

## Caveats

1. Dev only; the hidden test set is untouched. The fusion weight is chosen on the
   same dev split; the LOOCV number (+0.0087) is the honest estimate.
2. iota has 3 dev queries, law 11, workplace 11, quant 10; travel's mxbai
   encoding ran at ~46 doc/s on 158k docs (finished, but it is the most expensive
   domain and the least stable single-domain estimate).
3. The CE configuration (title query, window 250, max pooling) was fixed earlier
   on bge-base and reused unchanged for mxbai — no tuning against mxbai dev.
4. gte-large could not be run; no claim is made about it.
5. Absolute numbers remain far below the 30+ reported by TEMPO for 4B–7B models
   on a different corpus (1,730 queries, v1.0, 1,654,055 docs); we do not compare
   across those splits.

Reproduce:

```bash
bash scripts/phase8_baselines.sh        # e5-base + mxbai dense
bash scripts/phase8_mxbai_rerank.sh     # CE on mxbai top-100
python scripts/analyze_mxbai.py         # 13-domain fusion tables
python scripts/fair_11dom_compare.py    # shared-11 comparison
bash scripts/phase8_train_mxbai.sh      # train dense + CE (gate experiment)
bash scripts/phase8_train_features.sh
python scripts/temporal_gate_fit.py --train ... --dev ... --out results/temporal_gate_dev.json
```
