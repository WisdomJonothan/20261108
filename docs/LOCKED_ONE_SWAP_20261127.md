# Official replay of the fixed one-swap policy

The [paired membership experiment](PAIRED_MEMBERSHIP_20261127.md) found positive estimates against the maintained ranking on both the 641-query development panel and the separate 242-query panel. The unrestricted policy previously lost utility on official dev when it dropped judged positives already in the ranking. The single-swap variant limits that opportunity for harm and keeps the rest of the served ranking intact.

## Lock before scoring

Use the archived reference and local-scale proposals on all 519 official dev questions. Apply the exact previously tested rule. Take the first document from the proposal that is absent from the reference. Replace the deepest reference document absent from the proposal, with rank one protected. Leave the reference intact when either side is absent. Only this configuration advances. Its two training estimates and domain omission checks pass the existing point-estimate gate. The historical public dev scores motivated studying membership changes, so this is a locked follow-up on previously used dev.

Materialize the reference and single-swap rankings and fingerprint them before opening the official dev qrels. Report the official domain-macro nDCG@10, paired interval, minimum leave-one-domain-out effect, and the numbers of helpful and harmful swaps. Use no fitting, alternate guard, or extra model scores. The reference value must reproduce 0.40674178096988606. A positive gain that survives omitting any one domain is the criterion for changing the maintained ranking.

## Outcome

The fixed one-swap policy scores **0.40547** on all 519 queries versus **0.40674** for the maintained system. The change is −0.00127 with a paired 95% query-bootstrap interval of [−0.00422, +0.00171]. The smallest leave-one-domain-out change is −0.00236. The rule makes 126 swaps, improves seven queries, and worsens eleven. The official reference remains in place. Locked rankings and organizer scores are saved with the frozen ranking hash. [Replay artifact](../results/analysis/locked_one_swap_official_20261127.json)
