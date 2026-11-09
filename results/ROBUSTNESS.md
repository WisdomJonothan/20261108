# Robustness of the ranking: small domains, iota, and the n>=19 subset

Every number is **domain-macro nDCG@10** on the 519 dev queries, recomputed from
the committed per-domain score JSONs (`results/aggregate_*_dev.json`).

| system | 13 domains (official) | 12 domains (no iota) | queries in domains with n>=19 | iota alone (n=3) |
|---|---:|---:|---:|---:|
| zero-shot CE alone | 0.2680 | 0.2438 | 0.2772 | 0.5575 |
| SFT CE alone | 0.2742 | 0.2511 | 0.2840 | 0.5520 |
| dense + zero-shot CE, RRF 1:1 (V1) | 0.3166 | 0.2782 | 0.2913 | 0.7771 |
| dense + zero-shot CE, z-sum 1:1 | 0.3249 | 0.2832 | 0.2985 | 0.8254 |
| dense + SFT CE (1 epoch), z-sum 1:1 | **0.3336** | **0.2918** | **0.3043** | 0.8344 |
| dense + SFT CE (2 epochs, best@ep1), z-sum 1:1 | **0.3581** | **0.3181** | **0.3267** | 0.8383 |

## What this rules out

* **"The gain is carried by iota."** Dropping iota changes the 1-epoch SFT-vs-zero-shot
  delta from +0.0087 to **+0.0086**; the 12-domain numbers move together
  (0.2918 vs 0.2832). Iota is a 3-query domain whose within-domain mean is
  noisy (0.83 vs 0.78), but it is not the source of the effect. For the 2-epoch
  arm the 12-domain delta is **+0.0349**, i.e. larger than the 13-domain +0.0332.
* **"The gain is carried by the long history domain or by tiny domains."**
  Restricting to the 8 domains with at least 19 queries (the conventional
  small-domain filter; `domains_n_ge_19` in the aggregate JSONs) keeps the ordering: SFT 0.3043 vs zero-shot
  0.2985, +0.0058.
* Direction is consistent across statistics: 13-domain, 12-domain and n>=19 all
  put the SFT system first, and the per-domain table at w=1 shows 10/13 domains
  improved (`results/WEIGHT_SWEEP.md`).

## What it does not rule out

* Per-domain deltas are uneven: cardano +0.0336 (n=15) and hsm +0.0182 (n=45)
  dominate; bitcoin is -0.0080 (n=30) and politics -0.0011 (n=45). With 13
  domains, one or two domains can still swing the macro; the domain-clustered
  bootstrap ([+0.0037,+0.0142]) is the interval that accounts for that.
* A single training seed. These intervals resample queries/domains; they say
  nothing about training-seed variance.
