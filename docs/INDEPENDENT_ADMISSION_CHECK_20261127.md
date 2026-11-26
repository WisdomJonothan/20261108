# Transfer check for replacement decisions

The RETECO admission audit shows two distinct questions: whether a candidate can reach the final list and whether it improves the list when it displaces an incumbent. Use archived BRIGHT rankings from a separate research group to compare two policies on exactly the same document pair. This tests transfer to a different retrieval collection and leaves RETECO labels and training data outside the experiment. The BRIGHT inputs are rank signals from different query formulations using BM25. No document-level textual entailment judgments are made. [BRIGHT benchmark](https://brightbenchmark.github.io/) · [ADORE run release](https://github.com/aminbigdeli/ADORE/tree/288e9897af910f18934adb924034318406ea680e)

## Fixed protocol before opening relevance judgments

- Use all seven BRIGHT collections with rankings in the pinned ADORE release. The baseline is RRF with offset 60 over the three Query2doc prompting variants and GenQR. Run roles are based on their published method identities.
- In both replacement arms, nominate the highest-ranked ADORE document outside the baseline top ten. Replace rank ten only if the ADORE nomination outranks that incumbent in ADORE's list. A document absent from the archived top ten comes after all listed documents. Treat two absences as a tie.
- For the entrant-only control, additionally require the entrant to appear in ThinkQE's top ten. The paired arm also requires ThinkQE to prefer the same entrant to the same incumbent. Rank, proposal, available lists, and single-replacement budget are shared.
- As a separate structural comparison, fuse all six archived lists. Compare equal list weights to giving the three Query2doc variants a total weight of one. The four method families then have equal weight, with GenQR, ADORE, and ThinkQE each counted once. The candidate union and input ranks are shared.
- Produce complete rankings and fingerprint the output before pulling the relevance judgments. Then use binary nDCG@10 and equal collection weighting. Report paired query-resampling intervals, the number of helped and harmed queries, and the smallest effect after omitting one collection. Require positive paired-arm effects against both the frozen and matched entrant-only rankings, including all collection omissions, to advance that rule. Report the full matched family-weight comparison regardless of its result.

The experiment concerns a fixed replay of external ranks. Its score is separate from the 13-domain RETECO metric and the full BRIGHT leaderboard.

The preparation stage locked 749 rankings across the seven collections. The SHA-256 of the exact JSON passed to scoring is `b15edac29c3fa351cdb1d73cd8014f1913c35f5523ad7eb4e5c0ea8660fee4ba`.

## Matched textual-evidence comparison inside RETECO

Use the same proposed single swaps as the previous paired-membership replay. Include the 155-, 109- and 242-query training-side panels because their common evidence grader's score shards are archived. An inventory without labels finds grades for both documents on 143 of 144 proposed edits. The 186 and 191 panels stay outside this comparison. All three panels have already been used in prior research and are diagnostic here.

Both new arms require cached grades for both proposed documents, plus a verified challenger quote with reasoned grade at least two. This is the existing evidence standard. For a document use the reasoned grade when verified and the direct grade otherwise, with the same rule on both sides. The paired arm additionally requires the challenger grade to exceed the incumbent grade. The earlier admission agreement adjusted the candidate choice and required multiple cross-encoder checks. This comparison holds the exact pair, slot, number of evidence calls, and the challenger criterion fixed. It isolates only the effect of reading the incumbent's grade. Include the original always-swap policy as a reference for how many possible edits the evidence criteria leave behind. [Earlier agreement experiment](TEMPORAL_CONFLICT_RERANKING_20261126.md#replace-the-absolute-margin-with-cross-encoder-agreement)

Score these policies on the training labels after producing all rankings. Compare the paired and challenger-only arms by signed changes in nDCG, query counts, paired intervals, and domain deletion. An advance would require positive macro and worst-domain-omission effects against both the frozen list and the challenger-only policy. The RETECO dev score is outside this procedure.

## Independent check of method-family weighting

After reading BRIGHT, carry forward the exact six-list weight contrast only. Before reading any BEIR qrels, fix all five BEIR collections with complete corresponding archived runs in the same pinned release. Use the same six methods, offset 60, top-ten lists, and family weights. Average linearly graded nDCG@10 over queries and then over collections. Require a positive gain overall and after dropping any one collection. No collection-specific weights or alternative settings are tried. This checks transfer of the observation across benchmark collections, with a shared run publisher and query-rewriting pipeline. [Graded nDCG calculation](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c)

The BEIR preparation stage locked 1,455 rankings. Their SHA-256 is `7f9f3419c0ceea854a051584240f32a0e90234e7cd52ebcba992598c36b2ba7e`.

## Estimate voting weights from incremental retrieval value

The earlier fixed family average lowered the RETECO training-side score. Counting independent sources equally addresses repeat votes, while relevance quality and complementary coverage also affect how much each source should contribute. Test a deterministic way to estimate those contributions using only training domains. Reuse the 506 topics above. Group the two evidence-grader views and average their reciprocal contributions. Keep first-stage and pointwise rankings as separate groups. For every nonempty subset of the three groups compute its mean nDCG@10 with equal group weights and the same candidates allowed by that subset. The empty subset has value zero. Do not use the final head rule when scoring incomplete source subsets.

For each held-out domain in turn, estimate every group's value on the remaining domains by averaging its marginal nDCG contribution over the six possible group inclusion orders. Use those nonnegative contributions as weights for the held-out rankings, with negative contributions floored at zero. Compare against weighting by the standalone quality of each group, uniform family weights, and the maintained RETECO baseline. All full policies use the same available document union, the same four original lists, and the same head rule. Neither held-out judgments nor BRIGHT and BEIR labels determine weights. No additional inference or hyperparameter search is performed. Positive macro and positive worst-domain-omission effects against both the maintained baseline and the standalone-quality control are required. These training-domain results remain exploratory because the cohorts have appeared in earlier studies. [Earlier correlated-vote comparison](TEMPORAL_CONFLICT_RERANKING_20261126.md#discount-two-votes-produced-by-the-same-checkpoint)

## Results

### Check both documents on the same proposed pair

On the 506 previously used RETECO training questions, the frozen rank scores 0.28712 and the fixed always-swap replay scores 0.28848. Checking the entrant's verified grade raises the frozen score to 0.28779 and permits 58 edits. Comparing both grades permits 49 edits and gives the exact same nDCG for each query as the entrant-only control. All nine edits suppressed by the incumbent check leave nDCG unchanged under the available judgments. The extra criterion fails its matched-control gate.

On the external BRIGHT replay, the frozen four-list reference scores 0.22743, the entrant-only arm 0.24482, and the paired arm 0.24247. The paired arm makes 329 edits versus 366 for entrant-only. Relative to entrant-only it improves one query, worsens ten, and loses 0.00235 [−0.00431, −0.00073] on the equal-domain metric. Checking the incumbent with these available rank signals keeps more existing documents and blocks more helpful swaps than harmful swaps.

### Balance vote weights across method families

| Archived run replay | Topics | Equal weight per list | Equal weight per family | Paired change | Positive collections |
| --- | ---: | ---: | ---: | ---: | ---: |
| BRIGHT subset | 749 | 0.27829 | 0.33377 | +0.05548 [0.04714, 0.06422] | 7/7 |
| BEIR subset | 1,455 | 0.54687 | 0.54773 | +0.00086 [−0.00280, 0.00444] | 3/5 |

The smallest gain after removing a BRIGHT collection is +0.04927. On BEIR, omitting SciFact changes the estimate to −0.00022, below the transfer criterion. Equal family weights improve this BRIGHT rank combination. The fixed allocation has a smaller effect on BEIR, with mixed collection outcomes. These contrasts measure the effect of reallocating weight in the same six rankings. Family overlap and the relative quality of the groups can both contribute to that effect.

### Estimate native weights without held-out domain judgments

| RETECO training-side replay on the same 506 questions | Domain-macro nDCG@10 |
| --- | ---: |
| Maintained reference | 0.28712 |
| Uniform family weights | 0.28125 |
| Weight by standalone retrieval quality | 0.28176 |
| Weight by marginal retrieval value | 0.28210 |

The three-group marginal weighting is +0.00034 [−0.00038, +0.00109] versus standalone-quality weighting, and −0.00503 [−0.00751, −0.00291] versus the maintained reference. The hold-out-domain calibration recovers part of the uniform-family loss. This training comparison fails the reference gate, so the method stays outside official dev. Repeat votes, source quality and how each source complements the others should be measured as separate questions.

## Replay

The artifact includes both hashed external rank files before scoring, per-query scores, all contrasts, and domain-specific weights from the native replay. The external comparison can be repeated by running the script with `prepare --rankings <new-path>`, then `score --rankings <same-path> --fingerprint <printed-sha256> --output <new-path>`. Pass `--benchmark BEIR` to both stages for the BEIR comparison. Inputs are pinned to the external publisher revision. The two native replays take `--archive-dir results/analysis --data-root data/track1_tempo --output <new-path>`.

- [External retrieval replay](../scripts/replay_bright_admission_transfer.py)
- [Cached textual-evidence replay](../scripts/replay_incumbent_evidence.py)
- [Training-domain fusion calibration](../scripts/replay_marginal_fusion_weights.py)
- [Rank files and per-query results](../results/analysis/independent-admission-check-20261127.tar.gz)
