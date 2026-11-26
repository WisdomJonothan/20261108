# Test a single top-ten membership change

The preceding locked experiment recovered relevant documents from the original candidate pool. It scored 0.40607 against 0.40674 on the 519-question official dev set. The nDCG loss from displaced incumbents exceeded the gain from admissions. Use the existing ranked lists to isolate membership changes from the rest of the reorder. [Locked comparison and decomposition](TEMPORAL_CONFLICT_RERANKING_20261126.md#locked-local-scale-comparison-on-all-519-official-dev-questions)

## Fixed comparison

For each frozen top ten, nominate the highest-ranked new entrant in the revised local-normalization list. Its counterpart is the deepest displaced result in the frozen ranks two through ten. Put the entrant into that exact slot. The other nine documents keep their rank and membership. This tests one actual replacement and its discounted effect, based on the same alternatives as the previous wider-candidate experiment.

The matched guard performs the same proposed replacement only when the entrant also appears and the incumbent also disappears under the archived alternative normalization. That alternative standardizes the same pointwise logits using the original 30 candidates. This tests stability of both sides of the replacement under normalization. The proposal, score caches, rank position, and candidate pools are identical in the guarded and always-swap arms. No additional model calls or training are required.

Screen the fixed rule on the existing four-panel development population of 641 training queries. Require positive macro change against the frozen rank, a positive estimate after omitting any domain, and positive gain over the matched always-swap arm before comparing on the separate 242. The 242 were used by earlier project experiments, including the preceding fixed policy. Keep the public dev comparison outside this screening procedure.

## Discovery and separate one-swap follow-up

The archived official result changed membership on 126 of 519 queries. It admitted eight judged-positive documents and displaced twelve. All twelve displaced positives were originally in ranks eight through ten. Of the remaining queries, 224 changed order with identical membership. These figures describe the previous locked run.

| Training-side policy on 641 queries | nDCG@10 | Change versus frozen |
| --- | ---: | ---: |
| Frozen reference | 0.31351 | 0 |
| Archived local-normalization proposal | 0.31497 | +0.00146 |
| At most one swap | 0.31444 | +0.00093 |
| At most one swap with stability guard | 0.31440 | +0.00089 |

The guard accepts 129 proposals, including eleven helpful and four harmful edits. It blocks 50, including three harmful edits and one helpful edit. Its paired macro contrast against always taking the same proposal is −0.00004 [−0.00092, +0.00065]. The guard ends at discovery under the matched-control gate.

The one-swap arm beats its frozen reference after any single domain omission, with a minimum of +0.00030, and after any panel omission, with a minimum of +0.00058. Its paired interval is [−0.00031, +0.00231]. Record a separate exploratory follow-up of this unchanged arm on the 242. Use the same exact proposal and isolated swap with no stability check. Score only the baseline and this policy there. Positive point estimates overall and after omitting any domain are required. This reuse of the 242 is a training-side comparison, while the locked public-dev reference remains separate.

## One-swap follow-up on 242 queries

The fixed one-swap policy scores **0.28068** versus **0.27935** for the frozen reference. Its gain is **+0.00134**, with a paired 95% interval of [−0.00009, +0.00333]. The minimum leave-one-domain-out gain is +0.00074. Of 74 proposed edits, five improve nDCG, three lower it, and 66 tie. The gain comes from Genealogy, History, and Travel; other domain means tie. This is a small training-side improvement on the separate panel. The result does not change the official reference.

Both nDCG results can be reproduced from the recorded ranking archives and local official training judgments. The following commands use a new output path for each result.

```text
python scripts/replay_paired_membership.py --archive-dir results/analysis --data-root data/track1_tempo --output paired-membership-discovery.json
python scripts/replay_paired_membership.py --archive-dir results/analysis --data-root data/track1_tempo --output paired-membership-confirmation.json --confirmation paired-membership-discovery.json --confirmation-policy one_swap
```

[Saved per-query rankings and paired comparisons](../results/analysis/paired-membership-20261127.tar.gz)
