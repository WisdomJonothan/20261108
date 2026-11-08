# Train-fitted temporal gate: pre-registered evaluation (dev touched once)

**Status: negative (no usable retrieval gain).** This is the protocol the reviewer
asked for: fit everything on `train`, choose the blending weight inside `train`,
then read `dev` exactly once.

## Protocol (frozen before the dev read)

* Rows: for every anchored train query, the dense top-100 candidates with
  features `ce` (CE of full query), `s_temp` (CE of the temporal statement built
  from `query_temporal_events` + `key_time_anchors` + `query_summary`), `gap`
  (distance between document year and the query interval, clipped), `has_year`.
  Labels are qrels.
* Head: logistic regression (standardised features), fitted on all train rows.
* Weight: `s = ce + λ · logit(p_time)`, λ ∈ {0.05, 0.1, 0.2, 0.4} chosen by
  query-grouped 5-fold CV **inside train** (best train CV: λ = 0.4, 0.3572 vs
  0.3564 for smaller λ).
* Dev: the frozen head and λ are applied once; nDCG@10 reported per domain and
  pooled, with query bootstrap and domain-clustered bootstrap.
* Pre-registered adoption threshold: ΔnDCG@10 ≥ +0.02 with CI lower bound > 0.

Train material: dense top-100 + CE runs for bitcoin, economics, politics,
genealogy, history, hsm, monero (`runs/dense_mxbai_*_train.trec`,
`runs/rerank_mxbai_*_train.trec`); 7 domains, scripts
`phase8_train_mxbai.sh` → `phase8_train_features.sh` → `temporal_gate_fit.py`.

## Dev result (152 anchored queries, 7 domains)

| domain | n | CE | gated | Δ |
|---|---|---|---|---|
| bitcoin | 6 | 0.2551 | 0.2069 | −0.0482 |
| economics | 7 | 0.2461 | 0.2442 | −0.0019 |
| genealogy | 13 | 0.4324 | 0.4338 | +0.0014 |
| history | 92 | 0.3388 | 0.3414 | +0.0025 |
| hsm | 9 | 0.5529 | 0.5533 | +0.0004 |
| monero | 3 | 0.3333 | 0.4297 | +0.0964 |
| politics | 22 | 0.5764 | 0.5921 | +0.0157 |
| **pooled** | **152** | **0.3862** | **0.3901** | **+0.0039** |

Pooled 95 % CI [−0.0023, +0.0103]; domain-clustered CI [−0.0006, +0.0142];
P(Δ>0) = 0.893. Effect is ~5× below the +0.02 threshold, and the only large
positive (monero +0.096) rests on 3 queries.

## Why the earlier +0.0245 disappeared

The optimistic number came from 5-fold CV **on dev**: the head saw dev labels in
training folds. When the head is fitted on train and dev is used once, the gain
falls from +0.0245 to +0.0039 and the CI stops excluding zero. This is the
expected selection effect, and it is the reason the threshold was pre-registered.

## Reading

* The temporal statement **is** discriminative in isolation (s_temp AUC 0.61–0.88,
  above the 0.5 placebo), consistent with `TEMPORAL_OBSERVABILITY.md`.
* On these corpora that discriminative signal **does not translate into ranking
  gains** over the plain cross-encoder, consistent with the year-consistency
  analysis in `TEMPORAL_CEILING.md` (many time-mismatched candidates are gold).
* Claim to make: "a train-fitted temporal gate gives no measurable nDCG@10 gain
  under a pre-registered protocol", not "temporal information is useless".

Reproduce:

```bash
bash scripts/phase8_train_mxbai.sh        # train dense + CE for 7 domains
bash scripts/phase8_train_features.sh     # train feature rows
python scripts/temporal_gate_fit.py \
  --train results/temporal_pairs_{bitcoin,economics,politics,genealogy,history,hsm,monero}_train.jsonl \
  --dev   results/temporal_pairs_{bitcoin,economics,politics,genealogy,history,hsm,monero}_dev.jsonl \
  --out results/temporal_gate_dev.json
```
