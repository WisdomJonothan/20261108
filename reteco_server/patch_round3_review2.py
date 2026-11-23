"""Append the 2026-11-24 review round to REPORT.md and README.md.

Text is built as whole paragraphs (one .format per paragraph) so no
implicit-concatenation-plus-percent bug can creep in.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def j(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main():
    nd = j("results/aggregate_ndcgfix_dev.json")["domain_macro"]
    fp = j("results/aggregate_fallp_dev.json")["domain_macro"]
    dv = j("results/analysis/diver_top20_65.json")
    lad = j("results/replacement_v2/oracle_ladder_dev.json")
    at = j("results/replacement_v2/loss_attribution_dev.json")
    sel = j("results/replacement_v2/selection_rules_dev.json")
    ss = j("results/analysis/step_support_si.json")["summary"]
    cnt = at["class_counts"]

    p1 = (
        "**16.1 The nDCG loss, with the pair mask fixed.** The mask in `train_ce.py` was "
        "`(rank_i < k) & (rank_j < k)`, which drops every pair that straddles the top-k "
        "boundary and gives zero gradient whenever no two positives sit inside the top ten "
        "together. It is now `|`. With the corrected mask, lambdarank nDCG@10 training on the "
        "969 train groups scores **{:.4f}** on dev; a possum-loss control trained on exactly "
        "the same groups, initialisation and budget scores **{:.4f}**, while the frozen "
        "checkpoint (1,211 queries, its own group set) reproduces 0.3582. The comparison is "
        "single-factor and the slate loss is **{:+.4f}**: the loss function is not what limits "
        "the cross-encoder at this data size."
    ).format(nd, fp, nd - fp)

    p2 = (
        "**16.2 A specialist group reranker on the same pool.** Diver-GroupRank-7B, run with "
        "its official template on the dense top-20 as one group (180-word excerpts, answer "
        "prefilled so the 7B does not spend the budget reasoning), scores {:.6f} against the "
        "dense order's {:.6f} over {} queries (five per domain): **{:+.4f} nDCG@10**, while "
        "moving seven more relevant documents into the top ten ({} -> {}) and repairing three "
        "zero-score queries ({} -> {}). The two-query-per-domain sample had suggested +0.08; "
        "that was a property of the sample, not of the model."
    ).format(dv["diver_macro"], dv["dense_macro"], dv["n_queries"],
             dv["diver_macro"] - dv["dense_macro"], dv["rel_top10_dense"],
             dv["rel_top10_diver"], dv["zero_dense"], dv["zero_diver"])

    p3 = (
        "**16.3 Where the 0.3735 configuration loses.** Replaying the frozen rule over the "
        "cached 18 judged actions reproduces {:.6f}. The ladder above it is: perfect selection "
        "among the actions the judge already accepted **{:.6f}**; perfect judgement and "
        "selection inside the six judged pairs **{:.6f}**; inside all eighteen **{:.6f}**. Of "
        "the {} queries that have a beneficial action among the six pairs, {} end positive, "
        "{} reject every beneficial action and {} accept one but pick a different action. Six "
        "alternative selection rules (prior, product, gain-weighted prior, slot-first, "
        "high-confidence prior) replayed over the same judgements reach at most {:.6f}, so no "
        "selection rule recovers the gap: the accept/reject decision loses the points."
    ).format(lad["current"], lad["select_accepted"], lad["perfect_six"],
             lad["perfect_eighteen"], lad["queries_with_beneficial_in_pairs"], cnt["positive"],
             cnt["missed_swap"], cnt["wrong_choice"],
             max(r["macro"] for r in sel["rules"].values()))

    p4 = (
        "**16.4 Step-level support judgement.** 220 (query, requirement, document) triples from "
        "the train split were judged in a single prompt: does the passage contain the specific "
        "evidence this requirement asks for. Gold documents score {:.3f} mean probability under "
        "their own requirement and {:.3f} under a sibling requirement of the same query "
        "(negatives from dense ranks 21-100: {:.3f}); the same documents under the global "
        "question (does this passage help answer the question) score {:.3f} against {:.3f}. "
        "Ranked AUC is 0.8156 for the requirement-conditioned judgement and 0.8025 for the "
        "global one, and 136 of the 220 gold documents are scored strictly higher under their "
        "own requirement than under a sibling. Requirement conditioning carries some signal, "
        "but at 4B it is worth about one AUC point over topical relevance, and the absolute "
        "level is low: the model says the gold document contains the required evidence only "
        "about a third of the time. That is consistent with 16.3 - the binding constraint is "
        "what the judge can verify, not which pair it is asked about."
    ).format(ss["pos_S_own"], ss["pos_S_other_step"], ss["neg_S_own"],
             ss["pos_Y_global"], ss["neg_Y_global"])

    header = "## 16. Review round 3 (2026-11-24): loss function, specialist reranker, judgement"
    section = "\n".join(["", header, "",
                         "Four questions from the review were answered with measurements; three "
                         "are negative, and the fourth localises the remaining loss.", "",
                         p1, "", p2, "", p3, "", p4, ""])

    report = ROOT / "results/replacement_v2/REPORT.md"
    txt = report.read_text(encoding="utf-8")
    if header in txt:
        print("REPORT: already present")
    else:
        i = txt.find("\n## 15.")
        txt = (txt[:i] + section + txt[i:]) if i > 0 else txt.rstrip() + "\n" + section
        report.write_text(txt, encoding="utf-8")
        print("REPORT: section 16 inserted")

    bullet = (
        "- **Review round 3 (2026-11-24):** nDCG loss with the mask fixed scores {:.4f} against "
        "a possum control on the same 969 groups {:.4f} (single-factor, so the loss is not the "
        "limit); Diver-GroupRank-7B on the dense top-20 is {:+.4f} nDCG@10 over 65 queries but "
        "moves +{} relevant documents into the top ten; the 0.3735 configuration loses to "
        "accept/reject ({} of {} queries with a reachable beneficial action reject them all) "
        "and no selection rule recovers it; step-level support judgement reaches AUC 0.816 "
        "against 0.803 for the global question."
    ).format(nd, fp, dv["diver_macro"] - dv["dense_macro"],
             dv["rel_top10_diver"] - dv["rel_top10_dense"], cnt["missed_swap"],
             cnt["missed_swap"] + cnt["wrong_choice"] + cnt["positive"])

    readme = ROOT / "README.md"
    rtxt = readme.read_text(encoding="utf-8")
    if "Review round 3 (2026-11-24)" in rtxt:
        print("README: already present")
    else:
        i = rtxt.find("\n## Paper")
        rtxt = (rtxt[:i] + "\n" + bullet + "\n" + rtxt[i:]) if i > 0 else rtxt.rstrip() + "\n" + bullet + "\n"
        readme.write_text(rtxt, encoding="utf-8")
        print("README: bullet inserted")


if __name__ == "__main__":
    main()
