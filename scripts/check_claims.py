"""Check the numbers printed in the paper against the artefacts that produced them.

Every claim is (label, paper value, extractor, source path, in_paper). The
extractor pulls a number out of a committed JSON/Markdown artefact; the check
passes when it rounds to the paper's 4-decimal value (or is within 2e-4, for
aggregations that differ by a rounding step). Claims with in_paper=True are
additionally checked against the current `docs/PAPER_4PAGE.md`, so a stale
paper fails the gate instead of silently disagreeing with the artefacts.
Exits non-zero if any claim fails.
"""
import json
import os
import sys

R = "results/"
PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "docs", "PAPER_4PAGE.md")


def jload(p):
    with open(R + p, encoding="utf-8") as fh:
        return json.load(fh)


def dom_macro(p):
    return jload(p)["domain_macro"]


def boot(p, key):
    return jload(p)[key]["mean"], jload(p)[key]["ci_lo"], jload(p)[key]["ci_hi"]


def probe(model, arm):
    return jload("CE_FUSION_PROBE_FIXED.json")["models"][model][arm]["macro"]


def md_number(p, needle):
    txt = open(R + p, encoding="utf-8").read()
    i = txt.index(needle)
    return float(txt[i:i + 200].split("|")[1].strip())


CLAIMS = [
    # --- frozen system, the headline positive result -------------------------
    ("frozen dev nDCG@10", 0.3581, lambda: dom_macro("aggregate_sub_sft_e2bc_dev.json"),
     "aggregate_sub_sft_e2bc_dev.json", True),
    # --- re-verification and bounded-replacement ceiling (2026-11-20) --------
    ("frozen dev nDCG@10, re-verified", 0.3581,
     lambda: jload("FROZEN_BASELINE_20261120.json")["new_reverification"]["domain_macro"],
     "FROZEN_BASELINE_20261120.json", False),
    ("bounded-replacement oracle (cap 3)", 0.3769,
     lambda: jload("replacement_v2/oracle_fixed_dev.json")["macro_oracle"],
     "replacement_v2/oracle_fixed_dev.json", False),
    # --- dev replacement study, 2026-11-20 (Direct/Context/per-doc/active) ---
    ("replacement Direct dev nDCG@10", 0.3646,
     lambda: jload("replacement_v2/exp_direct_context_dev.json")["policies"]["direct"]["macro_policy"],
     "replacement_v2/exp_direct_context_dev.json", False),
    ("replacement Direct gain", 0.0065,
     lambda: jload("replacement_v2/exp_direct_context_dev.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/exp_direct_context_dev.json", False),
    ("replacement Context dev nDCG@10", 0.3632,
     lambda: jload("replacement_v2/exp_direct_context_dev.json")["policies"]["contextual"]["macro_policy"],
     "replacement_v2/exp_direct_context_dev.json", False),
    ("replacement Context gain", 0.0051,
     lambda: jload("replacement_v2/exp_direct_context_dev.json")["policies"]["contextual"]["macro_delta"],
     "replacement_v2/exp_direct_context_dev.json", False),
    ("replacement per-doc dev nDCG@10", 0.3598,
     lambda: jload("replacement_v2/exp_perdoc_dev.json")["policies"]["perdoc"]["macro_policy"],
     "replacement_v2/exp_perdoc_dev.json", False),
    ("replacement active-PRP dev nDCG@10", 0.3626,
     lambda: jload("replacement_v2/exp_active_dev.json")["policies"]["active"]["macro_policy"],
     "replacement_v2/exp_active_dev.json", False),
    # --- dev replacement study, extended 2026-11-20 (controls, budget, gate) --
    ("Direct CI low", 0.0021,
     lambda: jload("replacement_v2/exp_direct_context_dev.json")["policies"]["direct"]["bootstrap"]["ci95"][0],
     "replacement_v2/exp_direct_context_dev.json", False),
    ("Direct CI high", 0.0117,
     lambda: jload("replacement_v2/exp_direct_context_dev.json")["policies"]["direct"]["bootstrap"]["ci95"][1],
     "replacement_v2/exp_direct_context_dev.json", False),
    ("Context CI low", 0.0013,
     lambda: jload("replacement_v2/exp_direct_context_dev.json")["policies"]["contextual"]["bootstrap"]["ci95"][0],
     "replacement_v2/exp_direct_context_dev.json", False),
    ("Direct minus Context, paired", -0.0015,
     lambda: jload("replacement_v2/compare_ctx_vs_direct.json")["delta_b_minus_a"],
     "replacement_v2/compare_ctx_vs_direct.json", False),
    ("budget cap 1 dev", 0.3610,
     lambda: jload("replacement_v2/exp_budget_cap1.json")["policies"]["direct"]["macro_policy"],
     "replacement_v2/exp_budget_cap1.json", False),
    ("budget cap 2 dev", 0.3655,
     lambda: jload("replacement_v2/exp_budget_cap2.json")["policies"]["direct"]["macro_policy"],
     "replacement_v2/exp_budget_cap2.json", False),
    ("ablation, no abstain", 0.3647,
     lambda: jload("replacement_v2/ablations_dev.json")["rules"]["no_abstain"]["macro_policy"],
     "replacement_v2/ablations_dev.json", False),
    ("ablation, forward order only", 0.3596,
     lambda: jload("replacement_v2/ablations_dev.json")["rules"]["no_bidir"]["macro_policy"],
     "replacement_v2/ablations_dev.json", False),
    ("Qwen3 reranker single swap", 0.3620,
     lambda: jload("replacement_v2/reranker_pool_dev.json")["modes"]["single_swap"]["macro_policy"],
     "replacement_v2/reranker_pool_dev.json", False),
    ("Qwen3 reranker full 13", 0.3049,
     lambda: jload("replacement_v2/reranker_pool_dev.json")["modes"]["full_rerank"]["macro_policy"],
     "replacement_v2/reranker_pool_dev.json", False),
    ("Qwen3 reranker dense-100", 0.2213,
     lambda: jload("replacement_v2/reranker_deep100_dev.json")["modes"]["deep100"]["macro_policy"],
     "replacement_v2/reranker_deep100_dev.json", False),
    ("gate threshold", 0.0549,
     lambda: jload("replacement_v2/gate_eval.json")["calibration"]["tau"],
     "replacement_v2/gate_eval.json", False),
    ("gate dev delta", 0.0065,
     lambda: jload("replacement_v2/gate_eval.json")["dev"]["gate_judge"]["macro_delta"],
     "replacement_v2/gate_eval.json", False),
    ("cheap gap rule dev delta", 0.0072,
     lambda: jload("replacement_v2/gate_eval.json")["dev"]["ce_or_dense_gap"]["macro_delta"],
     "replacement_v2/gate_eval.json", False),
    ("precall gate dev delta", 0.0011,
     lambda: jload("replacement_v2/gate_eval.json")["dev"]["precall_cheap_gate"]["macro_delta"],
     "replacement_v2/gate_eval.json", False),
    ("fold f0 delta", 0.0007,
     lambda: jload("replacement_v2/exp_train_f0.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/exp_train_f0.json", False),
    ("fold f1 delta", 0.0032,
     lambda: jload("replacement_v2/exp_train_f1.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/exp_train_f1.json", False),
    ("fold f2 delta", 0.0068,
     lambda: jload("replacement_v2/exp_train_f2.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/exp_train_f2.json", False),
    ("reranker-start replacement nDCG@10", 0.2696,
     lambda: jload("replacement_v2/exp_rr20_dev.json")["policies"]["direct"]["macro_policy"],
     "replacement_v2/exp_rr20_dev.json", False),
    ("reranker-start gain", 0.0131,
     lambda: jload("replacement_v2/exp_rr20_dev.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/exp_rr20_dev.json", False),
    ("FollowIR Core17 frozen, original instructions", 0.2674,
     lambda: jload("replacement_v2/followir/scores_followir_core17.json")["graded_ndcg10"]["fo_og"]["frozen"],
     "replacement_v2/followir/scores_followir_core17.json", False),
    ("FollowIR Core17 frozen, changed instructions", 0.1761,
     lambda: jload("replacement_v2/followir/scores_followir_core17.json")["graded_ndcg10"]["fo_chg"]["frozen"],
     "replacement_v2/followir/scores_followir_core17.json", False),
    ("FollowIR News21 frozen fusion (og)", 0.3401,
     lambda: jload("replacement_v2/followir/scores_followir.json")["collections"]["news21"]["graded_ndcg10"]["fn21_og"]["frozen"],
     "replacement_v2/followir/scores_followir.json", False),
    ("FollowIR Robust04 frozen fusion (og)", 0.3092,
     lambda: jload("replacement_v2/followir/scores_followir.json")["collections"]["robust04"]["graded_ndcg10"]["frob_og"]["frozen"],
     "replacement_v2/followir/scores_followir.json", False),
    ("FollowIR Robust04 Direct gain", 0.0064,
     lambda: jload("replacement_v2/followir/exp_frob_dev.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/followir/exp_frob_dev.json", False),
    ("FollowIR News21 Direct gain", 0.0072,
     lambda: jload("replacement_v2/followir/exp_fn21_dev.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/followir/exp_fn21_dev.json", False),
    ("fair reranker single_swap nDCG@10", 0.3616,
     lambda: jload("replacement_v2/reranker_fair_dev.domainmacro.json")["modes"]["single_swap"]["macro_policy"],
     "replacement_v2/reranker_fair_dev.domainmacro.json", False),
    ("fair reranker single_swap delta", 0.0035,
     lambda: jload("replacement_v2/reranker_fair_dev.domainmacro.json")["modes"]["single_swap"]["macro_delta"],
     "replacement_v2/reranker_fair_dev.domainmacro.json", False),
    ("fair reranker full_rerank delta", -0.0408,
     lambda: jload("replacement_v2/reranker_fair_dev.domainmacro.json")["modes"]["full_rerank"]["macro_delta"],
     "replacement_v2/reranker_fair_dev.domainmacro.json", False),
    ("p-MRR Core17 frozen", -0.0433,
     lambda: jload("replacement_v2/followir/pmrr.json")["core17"]["frozen"]["p_mrr"],
     "replacement_v2/followir/pmrr.json", False),
    ("p-MRR News21 frozen", -0.0208,
     lambda: jload("replacement_v2/followir/pmrr.json")["news21"]["frozen"]["p_mrr"],
     "replacement_v2/followir/pmrr.json", False),
    ("p-MRR Robust04 frozen", -0.0347,
     lambda: jload("replacement_v2/followir/pmrr.json")["robust04"]["frozen"]["p_mrr"],
     "replacement_v2/followir/pmrr.json", False),
    ("error analysis fixable slots", 2497,
     lambda: jload("replacement_v2/error_analysis_dev.json")["wrong_slots_with_relevant_in_dense100"],
     "replacement_v2/error_analysis_dev.json", False),
    ("error analysis gain share ranks 1-3", 0.4361,
     lambda: jload("replacement_v2/error_analysis_dev.json")["gain_share_by_band"]["1-3"],
     "replacement_v2/error_analysis_dev.json", False),
    ("dense-start replacement nDCG@10", 0.3049,
     lambda: jload("replacement_v2/exp_dense_start_dev.json")["policies"]["direct"]["macro_policy"],
     "replacement_v2/exp_dense_start_dev.json", False),
    ("dense-start replacement gain", 0.0106,
     lambda: jload("replacement_v2/exp_dense_start_dev.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/exp_dense_start_dev.json", False),
    ("calibration split delta", -0.0014,
     lambda: jload("replacement_v2/exp_train_fall.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/exp_train_fall.json", False),
    ("query reasoning orig", 0.2933,
     lambda: jload("replacement_v2/query_reason_summary.json")["orig"]["mean"],
     "replacement_v2/query_reason_summary.json", False),
    ("query reasoning with analysis", 0.2930,
     lambda: jload("replacement_v2/query_reason_summary.json")["reason"]["mean"],
     "replacement_v2/query_reason_summary.json", False),
    ("query reasoning analysis only", 0.2546,
     lambda: jload("replacement_v2/query_reason_summary.json")["analysis"]["mean"],
     "replacement_v2/query_reason_summary.json", False),
    ("violation filter swaps", 3,
     lambda: sum(jload("replacement_v2/exp_violation_dev.json")["swap_counts"][k]
                 for k in ("helpful", "harmful", "neutral")),
     "replacement_v2/exp_violation_dev.json", False),
    ("violation filter dev macro", 0.358099,
     lambda: jload("replacement_v2/exp_violation_dev.json")["macro_policy"],
     "replacement_v2/exp_violation_dev.json", False),
    ("frozen gain vs zero-shot", 0.0332,
     lambda: boot("boot_dom_e2_vs_zs.json", "domain_bootstrap")[0], "boot_dom_e2_vs_zs.json", True),
    ("frozen CI low", 0.0192,
     lambda: boot("boot_dom_e2_vs_zs.json", "domain_bootstrap")[1], "boot_dom_e2_vs_zs.json", True),
    ("frozen CI high", 0.0472,
     lambda: boot("boot_dom_e2_vs_zs.json", "domain_bootstrap")[2], "boot_dom_e2_vs_zs.json", True),
    ("dense dev nDCG@10", 0.2943, lambda: dom_macro("aggregate_dense_mxbai_dev.json"),
     "aggregate_dense_mxbai_dev.json", True),
    ("zero-shot CE dev", 0.3249, lambda: dom_macro("aggregate_sub_zs11_dev.json"),
     "aggregate_sub_zs11_dev.json", True),
    # --- Phase 17 ladder, after the query-split and cue-regex fixes ----------
    ("same-code retrain dev", 0.3575, lambda: dom_macro("aggregate_sub_lx_possum2_dev.json"),
     "aggregate_sub_lx_possum2_dev.json", True),
    ("step arm, weight 0.3 dev", 0.3593, lambda: dom_macro("aggregate_sub_sx_l03b_dev.json"),
     "aggregate_sub_sx_l03b_dev.json", True),
    ("step arm, weight 1.0 dev", 0.3600, lambda: dom_macro("aggregate_sub_sx_l10b_dev.json"),
     "aggregate_sub_sx_l10b_dev.json", True),
    ("Qwen3 CE alone", 0.2313, lambda: dom_macro("aggregate_ce_qwen3_dev.json"),
     "aggregate_ce_qwen3_dev.json", True),
    ("Qwen3 fused 1:1", 0.2996, lambda: dom_macro("aggregate_sub_qwen3_dev.json"),
     "aggregate_sub_qwen3_dev.json", True),
    ("backfill arm dev", 0.3565, lambda: dom_macro("aggregate_sub_cx_bfb_dev.json"),
     "aggregate_sub_cx_bfb_dev.json", True),
    ("full-question arm dev", 0.3443, lambda: dom_macro("aggregate_sub_cx_qv_dev.json"),
     "aggregate_sub_cx_qv_dev.json", True),
    ("deployment-budget epoch dev", 0.3560, lambda: dom_macro("aggregate_sub_lx_ep1b_dev.json"),
     "aggregate_sub_lx_ep1b_dev.json", True),
    # --- the cross-encoder / fusion inversion --------------------------------
    ("cx_qv fused at best w", 0.3546, lambda: probe("cx_qv", "w=2"),
     "CE_FUSION_PROBE_FIXED.json", True),
    ("cx_qv cross-encoder alone", 0.3152, lambda: probe("cx_qv", "ce_only"),
     "CE_FUSION_PROBE_FIXED.json", True),
    ("lx_possum cross-encoder alone", 0.2765, lambda: probe("lx_possum2", "ce_only"),
     "CE_FUSION_PROBE_FIXED.json", True),
    ("cx_qv fusion at w=1", 0.3443, lambda: probe("cx_qv", "w=1"),
     "CE_FUSION_PROBE_FIXED.json", True),
    ("lx_possum fusion at w=1", 0.3575, lambda: probe("lx_possum2", "w=1"),
     "CE_FUSION_PROBE_FIXED.json", True),
    ("lx_possum fusion at val weight", 0.3522, lambda: probe("lx_possum2", "w=0.75"),
     "CE_FUSION_PROBE_FIXED.json", True),
    ("cx_qv fusion at val weight", 0.3509, lambda: probe("cx_qv", "w=1.25"),
     "CE_FUSION_PROBE_FIXED.json", True),
    # --- diagnostics quoted in the error analysis ----------------------------
    ("coverage bound, fused", 1.3405, lambda: dom_macro("COVERAGE_CF_BOUND.json")["cov_cf"],
     "COVERAGE_CF_BOUND.json", False),
    ("coverage bound, CE alone", 1.1844, lambda: dom_macro("COVERAGE_CF_BOUND_ce.json")["cov_cf"],
     "COVERAGE_CF_BOUND_ce.json", False),
    # Not our computation: the organisers' published Track 1a dev baseline.
    # --- widened action space, corrected harness (2026-11-22) ----------------
    ("widened fused k=2 delta", 0.008242,
     lambda: jload("replacement_v2/expanded3_fixed_dev.json")["budgets"]["2"]["macro_delta"],
     "replacement_v2/expanded3_fixed_dev.json", False),
    ("widened fused k=6 delta", 0.012151,
     lambda: jload("replacement_v2/expanded3_fixed_dev.json")["budgets"]["6"]["macro_delta"],
     "replacement_v2/expanded3_fixed_dev.json", False),
    ("widened fused k=18 delta", 0.011239,
     lambda: jload("replacement_v2/expanded3_fixed_dev.json")["budgets"]["18"]["macro_delta"],
     "replacement_v2/expanded3_fixed_dev.json", False),
    ("widened dense k=2 delta", 0.010847,
     lambda: jload("replacement_v2/expanded3_fixed_densestart_dev.json")["budgets"]["2"]["macro_delta"],
     "replacement_v2/expanded3_fixed_densestart_dev.json", False),
    ("widened dense k=6 delta", 0.023219,
     lambda: jload("replacement_v2/expanded3_fixed_densestart_dev.json")["budgets"]["6"]["macro_delta"],
     "replacement_v2/expanded3_fixed_densestart_dev.json", False),
    ("random-subset control k=18", -0.003608,
     lambda: jload("replacement_v2/expanded3_fixed_random_dev.json")["budgets"]["18"]["macro_delta"],
     "replacement_v2/expanded3_fixed_random_dev.json", False),
    ("replay uncertainty k=2", -0.000454,
     lambda: jload("replacement_v2/expanded3_fixed_policies.json")["budgets"]["2"]["uncertainty"]["delta"],
     "replacement_v2/expanded3_fixed_policies.json", False),
    ("replay random-in-set k=6", 0.013198,
     lambda: jload("replacement_v2/expanded3_fixed_policies.json")["budgets"]["6"]["random"]["delta"],
     "replacement_v2/expanded3_fixed_policies.json", False),
    ("replay oracle ordering k=2", 0.025754,
     lambda: jload("replacement_v2/expanded3_fixed_policies.json")["budgets"]["2"]["oracle_order"]["delta"],
     "replacement_v2/expanded3_fixed_policies.json", False),
    ("condition-pair judge dev", 0.3625,
     lambda: jload("replacement_v2/exp_violation_pair_dev.json")["macro_policy"],
     "replacement_v2/exp_violation_pair_dev.json", False),
    ("top-slot window nDCG@10", 0.353916,
     lambda: jload("replacement_v2/exp_direct_top3_dev.json")["policies"]["direct"]["macro_policy"],
     "replacement_v2/exp_direct_top3_dev.json", False),
    ("top-slot window helpful swaps", 23,
     lambda: jload("replacement_v2/exp_direct_top3_dev.json")["policies"]["direct"]["swap_counts"]["helpful"],
     "replacement_v2/exp_direct_top3_dev.json", False),
    ("condition arm Core17 delta", 0.0,
     lambda: jload("replacement_v2/followir/exp_cond_followir_dev.json")["macro_delta"],
     "replacement_v2/followir/exp_cond_followir_dev.json", False),
    ("condition arm News21 delta", 0.001889,
     lambda: jload("replacement_v2/followir/exp_cond_fn21_dev.json")["macro_delta"],
     "replacement_v2/followir/exp_cond_fn21_dev.json", False),
    ("condition arm Robust04 delta", 0.001891,
     lambda: jload("replacement_v2/followir/exp_cond_frob_dev.json")["macro_delta"],
     "replacement_v2/followir/exp_cond_frob_dev.json", False),
    ("learned gain model dev delta", 0.003381,
     lambda: jload("replacement_v2/gain_model_bounded_dev.json")["macro_delta"],
     "replacement_v2/gain_model_bounded_dev.json", False),
    ("learned gain model train AUC", 0.6621,
     lambda: jload("replacement_v2/gain_model_bounded_dev.json")["train_auc"],
     "replacement_v2/gain_model_bounded_dev.json", False),
    ("held-out fixable slots", 5161,
     lambda: jload("replacement_v2/error_analysis_train.json")["wrong_fixable"],
     "replacement_v2/error_analysis_train.json", False),
    ("held-out gain share ranks 1-3", 0.4469,
     lambda: jload("replacement_v2/error_analysis_train.json")["gain_share"]["1-3"],
     "replacement_v2/error_analysis_train.json", False),
    ("ensemble both-agree delta", 0.003135,
     lambda: jload("replacement_v2/ensemble_rules_dev.json")["macro_delta"],
     "replacement_v2/ensemble_rules_dev.json", False),
    ("6/3 split nDCG@10", 0.364672,
     lambda: jload("replacement_v2/split_k6.json")["macro_policy"],
     "replacement_v2/split_k6.json", False),
    ("6/3 split harmful swaps", 13,
     lambda: jload("replacement_v2/split_k6.json")["harmful"],
     "replacement_v2/split_k6.json", False),
    ("evidence-first sample delta", -0.000638,
     lambda: jload("replacement_v2/exp_evidence_sample_dev.json")["macro_delta"],
     "replacement_v2/exp_evidence_sample_dev.json", False),
    ("cascade swaps on uncertain subset", 13,
     lambda: jload("replacement_v2/exp_violation_cascade_dev.json")["swap_counts"]["neutral"],
     "replacement_v2/exp_violation_cascade_dev.json", False),
    ("combined prompt nDCG@10", 0.362638,
     lambda: jload("replacement_v2/exp_combined_pair_dev.json")["macro_policy"],
     "replacement_v2/exp_combined_pair_dev.json", False),
    ("combined prompt helpful swaps", 31,
     lambda: jload("replacement_v2/exp_combined_pair_dev.json")["swap_counts"]["helpful"],
     "replacement_v2/exp_combined_pair_dev.json", False),
    ("condition-pair judge, dense start", 0.3014,
     lambda: jload("replacement_v2/exp_violation_pair_dense_dev.json")["macro_policy"],
     "replacement_v2/exp_violation_pair_dense_dev.json", False),
    ("condition-pair judge, reranker start", 0.2598,
     lambda: jload("replacement_v2/exp_violation_pair_rr_dev.json")["macro_policy"],
     "replacement_v2/exp_violation_pair_rr_dev.json", False),
    ("combined condition+relevance rule", 0.3661,
     lambda: jload("replacement_v2/hybrid_rules_dev.json")["either"]["macro"],
     "replacement_v2/hybrid_rules_dev.json", False),
    ("BM25 baseline (organisers)", 0.1147,
     lambda: float(open(R + "PHASE9_PLAN.md", encoding="utf-8").read()
                   .split("official BM25 1a dev baseline (organizers)")[1]
                   .split("|")[1].strip()),
     "PHASE9_PLAN.md (organisers' data page)", True),
    # --- p-MRR for the judged arms and the wider evidence window (2026-11-22) --
    ("p-MRR Core17 Direct", -0.0055,
     lambda: jload("replacement_v2/followir/pmrr_arms.json")["core17"]["direct"]["p_mrr"],
     "replacement_v2/followir/pmrr_arms.json", False),
    ("p-MRR Core17 condition", -0.0051,
     lambda: jload("replacement_v2/followir/pmrr_arms.json")["core17"]["cond"]["p_mrr"],
     "replacement_v2/followir/pmrr_arms.json", False),
    ("p-MRR Robust04 Direct", -0.0121,
     lambda: jload("replacement_v2/followir/pmrr_arms.json")["robust04"]["direct"]["p_mrr"],
     "replacement_v2/followir/pmrr_arms.json", False),
    ("p-MRR News21 condition", -0.0111,
     lambda: jload("replacement_v2/followir/pmrr_arms.json")["news21"]["cond"]["p_mrr"],
     "replacement_v2/followir/pmrr_arms.json", False),
    ("wide-window dev nDCG@10", 0.3660,
     lambda: jload("replacement_v2/exp_wide_win400_dev.json")["policies"]["direct"]["macro_policy"],
     "replacement_v2/exp_wide_win400_dev.json", False),
    ("wide-window gain", 0.0079,
     lambda: jload("replacement_v2/exp_wide_win400_dev.json")["policies"]["direct"]["macro_delta"],
     "replacement_v2/exp_wide_win400_dev.json", False),

]


def main():
    paper = open(PAPER, encoding="utf-8").read() if os.path.isfile(PAPER) else ""
    ok = True
    rows = []
    for label, expected, fn, src, in_paper in CLAIMS:
        try:
            got = float(fn())
        except Exception as e:  # noqa: BLE001
            rows.append((label, expected, None, src, "ERROR %s" % e))
            ok = False
            continue
        close = abs(round(got, 4) - expected) <= 2e-4
        status = "ok" if close else "MISMATCH"
        ok = ok and close
        if in_paper:
            # the paper prints four decimals; the table cells too
            if ("%.4f" % expected) not in paper:
                status += " / not-in-paper"
                ok = False
        rows.append((label, expected, got, src, status))
    w = max(len(r[0]) for r in rows)
    for label, expected, got, src, status in rows:
        g = "n/a" if got is None else "%.4f" % got
        print("%-*s paper %.4f  artefact %s  %-9s %s" % (w, label, expected, g, status, src))
    print("\nRESULT:", "all claims traceable" if ok else "CLAIM MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
