# Sanity controls for the fusion claim

Question: is the +0.0332 gain an artefact of the z-score fusion algebra, or does it
require real cross-encoder scores?

Control: keep every candidate doc, but permute the cross-encoder scores (two variants),
re-fuse with the frozen dense run at the frozen z 1:1 settings, and re-score.

| variant | what is permuted | domain-macro nDCG@10 |
|---|---|---|
| dense alone | — | 0.2943 |
| frozen system (SFT CE) | — | 0.3581 |
| permuted CE, whole pool | all 1000 doc scores of each topic | 0.2287 |
| permuted CE, in-pool only | scores of the 100 dense candidates only | 0.2314 |

Reading: replacing the cross-encoder's ordering by a random one drops the fused system
to ~0.23, i.e. *below* dense alone — the fused ordering is dominated by whichever signal
carries the ranking information, so the fusion weight cannot manufacture the gain. The
result also shows that the fusion is not "dense order + a small nudge": equal z weights
let a useless CE overwrite a good dense order, which is why the score is below 0.2943.

Files: `runs/perm_sft_e2bc_*_dev.trec`, `runs/fused_perm_*_dev.trec`,
`results/score_perm_*_dev.json`, `results/aggregate_perm_dev.json` (whole pool),
`results/aggregate_perm2_dev.json` (in-pool).  Reproduce: `scripts/fuse_scores.py --norm z
--wa 1 --wb 1` on dense + permuted CE, then `score_run.py` + `aggregate_scores.py`.
