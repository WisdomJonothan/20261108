#!/usr/bin/env python
"""Assemble results/analysis/SCORE_DIAGNOSIS.json from the three measurement passes."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "analysis")


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return json.load(f)


core = load("SCORE_DIAGNOSIS.json")          # pass 1 (rewritten here with everything)
extra = load("SCORE_DIAGNOSIS_EXTRA.json")   # pass 2
spec = load("SCORE_DIAGNOSIS_SPECIALISATION.json")  # pass 3

BASE = core["meta"]["frozen_baseline_macro"]
POOL = extra["pool_oracle"]["macro_ideal_rerank_of_frozen_pool100"]
DENSE = extra["depth_curves"]["dense_actual_k10"]
DEPTH = extra["depth_curves"]["dense_oracle_by_depth"]
ORDER = extra["ordering_only"]["macro_if_frozen_top10_perfectly_reordered"]
ZC = extra["pool_oracle"]
SP = spec

levers = {
    "ranking_rule": "expected dev points = (achievable ceiling) x (capture rate supported by a "
                    "measured experiment); ceilings are oracle numbers and are never counted as expected gain",
    "items": [
        {
            "rank": 1, "lever": "C. better ranking of the documents already in the frozen top-100 pool",
            "ceiling_points": round(100 * (POOL - BASE), 2),
            "ceiling_detail": f"pool-oracle nDCG@10 {POOL:.4f} vs frozen {BASE:.4f}; "
                              f"perfect ordering of the same 10 returned docs alone is worth "
                              f"{100*(ORDER-BASE):.2f} pts (296/519 queries), so "
                              f"{100*(ORDER-BASE)/(POOL-BASE):.1f}% of the pool headroom is pure position error",
            "evidence": "dense 0.2943 -> frozen 0.3581 captures 14.2% of the dense->pool gap "
                        "(results/CAPTURE_CURVE_E2.md); CE-alone (full-query, seed 17) 0.3310 captures 8.2%; "
                        "best measured policy 0.3646 (Direct, +0.65 pts, CI [+0.21,+1.17])",
            "expected_points": [2.5, 5.8],
            "expected_detail": "raises capture from the measured 14.2% to 22-30% of the 45.03-pt "
                               "dense->pool gap: 0.2943+0.22*45.03=0.3934 (+3.53 pts) to "
                               "0.2943+0.30*45.03=0.4294 (+7.13 pts); a policy-only fix is worth "
                               "+0.5 to +1.0",
        },
        {
            "rank": 2, "lever": "A. promote the zero-class documents that are already in the pool",
            "ceiling_points": round(ZC["zero_class_oracle_points"], 2),
            "ceiling_detail": f"{ZC['n_zero']} zeros, {ZC['n_zero_with_pool_gold']} of them have a judged-relevant "
                              f"doc inside the delivered top-100; lifting them to their in-pool oracle is worth "
                              f"{ZC['zero_class_oracle_points']:.2f} pts "
                              f"({100*ZC['share_of_pool_headroom_in_zero_class']:.1f}% of the pool headroom). "
                              f"Lifting them to the domain median instead = {ZC['zero_class_median_points']:.2f} pts",
            "evidence": "519/519 queries have identical dense-top100 and frozen Rec@100, so the pool is the dense "
                        "top-100; the SFT CE reranking from the pool directly scored 0.3049 (-5.3 pts), i.e. the "
                        "current CE cannot promote them",
            "expected_points": [0.3, 1.5],
            "expected_detail": "same mechanism as C, so not additive; a targeted promotion policy at 10-20% "
                               "precision on 86 queries: 0.10*14.33=+1.43, 0.05*14.33=+0.72",
        },
        {
            "rank": 3, "lever": "E. per-domain specialisation (checkpoint and fusion weight)",
            "ceiling_points": round(SP["per_domain_oracle_over_arm_and_weight"]["macro"] - BASE and
                                    100 * (SP["per_domain_oracle_over_arm_and_weight"]["macro"] - BASE), 2),
            "ceiling_detail": f"in-sample per-domain oracle over 5 CE arms x 9 fusion weights = "
                              f"{SP['per_domain_oracle_over_arm_and_weight']['macro']:.4f}; weight-only "
                              f"{SP['per_domain_oracle_over_weight_only']['macro']:.4f}; arm-only "
                              f"{SP['per_domain_oracle_over_arm_only']['macro']:.4f}; single global best "
                              f"({SP['global_best']['arm']}, w={SP['global_best']['w_ce']}) "
                              f"{SP['global_best']['macro']:.4f}",
            "evidence": "committed results/weight_sweep_e2_dev.json: w=1.0 -> 0.3581, w=1.5 -> 0.3592 (+0.11); "
                        "per-domain selection is fitted on 3-240 queries per domain, so shrink 60-70%",
            "expected_points": [0.5, 1.0],
            "expected_detail": "0.35 x 2.32 = +0.81 (weight+arm), 0.35 x 1.19 = +0.42 (weight only)",
        },
        {
            "rank": 4, "lever": "D. more replacement swaps at the tail",
            "ceiling_points": 1.88,
            "ceiling_detail": "0.37694 (up to 3 tail replacements from union(dense20,CE20)) vs 0.3581; "
                              "one unlimited swap restricted to ranks 8-10 = 0.40975, any slot = 0.51036",
            "evidence": "error_analysis_dev.json: 43.6% of single-fix gain sits at ranks 1-3, only 21.4% at "
                        "8-10, so a tail-only policy is structurally capped at about a fifth of the replacement "
                        "headroom; measured capture of the 1.88-pt ceiling is 34.6% (+0.65 pts)",
            "expected_points": [0.2, 0.6],
            "expected_detail": "extending the judge to ranks 1-7 would move the ceiling from 1.88 to "
                               "100*(0.51036-BASE)=15.23 pts, but the decision quality measured so far "
                               "(1 helpful per 1.2 swaps) implies +0.2 to +0.6",
        },
        {
            "rank": 5, "lever": "B. first-stage recall (deeper or additional retrieval)",
            "ceiling_points": round(100 * (DEPTH["1000"] - DEPTH["100"]), 2),
            "ceiling_detail": f"dense@100 pool oracle {DEPTH['100']:.4f} -> dense@1000 {DEPTH['1000']:.4f}; "
                              f"22.5% of dev gold sits between dense rank 100 and 1000; only 10 of the "
                              f"138 zero queries have no relevant document anywhere in the dense top-1000",
            "evidence": "measured depth probe (62 queries, 200-deep CE pool): +0.0010 [-0.0009,+0.0036], "
                        "3 up / 2 down, zero in 9 of 13 domains; BM25 arms on the only covered domain have "
                        "lower oracle nDCG@10 than dense (0.5463-0.5822 vs 0.6349) and recover at most 1 of "
                        "the 4 dense zeros",
            "expected_points": [0.0, 0.5],
            "expected_detail": "0.5% of 19.71 pts (measured probe capture) = +0.10; a hybrid stage that "
                               "adds only new documents cannot pay unless the CE is retrained on them",
        },
    ],
    "largest_ceiling": {
        "lever": "C (equivalently: ranking, not retrieval)",
        "ceiling_points": round(100 * (POOL - BASE), 2),
        "share_of_remaining_metric": round((POOL - BASE) / (1.0 - BASE), 4),
        "realistic_capture": [0.20, 0.35],
        "realistic_points": [round(100 * 0.20 * (POOL - BASE), 2), round(100 * 0.35 * (POOL - BASE), 2)],
        "why": "the frozen system already sits on top of a pool whose oracle is 0.7446; the documents are "
               "there and half the recoverable nDCG is mis-ordering of the ten docs actually returned. "
               "Capture above 14.2% has never been demonstrated by any reranker in the tree "
               "(CE-alone 0.3310, direct deep rerank 0.2213, full pool rerank 0.3049), so 20-35% is the "
               "honest band, not the 60-100% an oracle suggests.",
    },
}

verification = {
    "confirmed": [
        "baseline 0.3580985233 recomputed from results/score_sub_sft_e2bc_<domain>_dev.json and "
        "independently from runs/sub_sft_e2bc_<domain>_dev_head10.trec; max per-query difference 0.0",
        "138/519 queries (26.59%) score exactly 0.0 - confirmed, and the per-domain split matches the "
        "earlier analysis (history 46, politics 14, bitcoin 13, travel 12, genealogy 10, hsm 10, "
        "economics 9, monero 8, quant 5, cardano 4, workplace 4, law 3, iota 0)",
        "519/519 queries are answerable (every qrels_dev query has >=1 positive) and every one of the "
        "1,777 unique (query, document) judged-relevant pairs was found in data/track1_tempo/<domain>/documents.jsonl "
        "on the server (0 missing), so the perfect domain-macro is 1.0",
        "the committed oracle ladders reproduce exactly: union(dense20,CE20) one swap any slot "
        "0.5103619641669876 vs committed 0.5103619641669878; ranks 8-10 0.4097536 vs the quoted 0.4098; "
        "dense@100 one swap 0.5833, two swaps 0.6299, full rerank 0.7446",
    ],
    "corrected": [
        "the brief's 'CE-alone 0.3249' is the zero-shot cross-encoder fused z 1:1 with dense "
        "(results/weight_sweep_dev.json summary zs=0.3249074); true CE-alone on dev is 0.29237 for title "
        "CE and 0.33096 for full-query CE (docs/RESEARCH_UPDATE_20261119.md table, reproduced here as "
        "0.33096)",
        "results/DEPTH_ANALYSIS.md prints recall@k in the oracle column for k=10/20/50 "
        "(0.3852/0.4844/0.6086); the official oracle-by-depth is 0.4437/0.5373/0.6607, i.e. the shallow "
        "pool is worth up to +5.9 pts more than that table claims. Its k>=100 rows are correct",
        "the earlier diagnosis 'the zero class is a first-stage recall failure (130 of 138 unreachable in "
        "dense top-100)' is wrong, and its artefact has a parse bug. reteco_server/zero_recall.py reads TREC "
        "rows as (docid=p[2], score=float(p[3])) but p[3] is the rank column, so it sorts each run by rank "
        "DESCENDING and then keeps the first 100 rows - for a 1000-deep dense run that slice is ranks "
        "901-1000, and inside a 100-deep CE run the reported index is 101-true_rank. Concretely, for "
        "bitcoin/112021_76 the gold doc is at dense rank 55 and CE rank 71, yet the artefact records dense "
        "best_rank=null and ce best_rank=30 (=101-71). Its frozen column reads runs/sub_sft_e2bc_<d>_dev.trec, "
        "which does not exist locally (only the _head10 prefix does), so that column is all null. The correct "
        "numbers: gold is inside the dense top-100 for 86 of the 138 zeros (128 inside the top-1000), and "
        "519/519 queries have identical dense-top100 and frozen Rec@100, i.e. the delivered pool holds the "
        "same relevant content as the dense top-100. 86 of the zeros are already inside the pool",
        "the earlier 'zeros -> domain non-zero mean = 0.472' is 0.4985 under the stated rule (each zero "
        "replaced by its own domain's non-zero mean, 13-domain macro); the domain-median variant asked "
        "for here is 0.4385",
    ],
}

out = {
    "generated_by": ["results/analysis/score_diagnosis.py",
                     "results/analysis/score_diagnosis_extra.py",
                     "results/analysis/score_diagnosis_specialisation.py",
                     "results/analysis/score_diagnosis_report.py"],
    "meta": core["meta"],
    "headline": {
        "frozen_dev_domain_macro": BASE,
        "n_queries": core["meta"]["n_queries"],
        "n_domains": 13,
        "n_zero_queries": ZC["n_zero"],
        "zero_share": ZC["n_zero"] / core["meta"]["n_queries"],
        "nanswerable": core["answerability"]["n_answerable"],
        "theoretical_perfect_macro": core["answerability"]["theoretical_perfect_macro"],
        "pool_oracle_100": POOL,
        "ordering_only_ceiling": ORDER,
        "dense_10_oracle": DEPTH["10"],
        "dense_1000_oracle": DEPTH["1000"],
    },
    "distribution": core["distribution"],
    "zero_class": core["zero_class"],
    "pool_oracle_decomposition": extra["pool_oracle"],
    "ordering_only": extra["ordering_only"],
    "per_domain": core["per_domain"],
    "answerability": core["answerability"],
    "depth_curve": core["depth_curve"],
    "depth_curves_local": extra["depth_curves"],
    "ce_actual": core["ce_actual"],
    "ladders": {
        "committed_and_reproduced": core["ladders"],
        "union20_ladder": extra["union_ladder"],
    },
    "specialisation": {
        "at_w1": SP["at_w1"],
        "global_best": SP["global_best"],
        "per_domain_oracle_over_arm_and_weight": SP["per_domain_oracle_over_arm_and_weight"],
        "per_domain_oracle_over_weight_only": SP["per_domain_oracle_over_weight_only"],
        "per_domain_oracle_over_arm_only": SP["per_domain_oracle_over_arm_only"],
    },
    "levers": levers,
    "verification": verification,
}
with open(os.path.join(OUT, "SCORE_DIAGNOSIS.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("wrote", os.path.join(OUT, "SCORE_DIAGNOSIS.json"))
print(json.dumps(out["headline"], indent=1))
for it in levers["items"]:
    print(f"{it['rank']}. {it['lever'][:60]:60s} ceiling {it['ceiling_points']:6.2f} "
          f"expected {it['expected_points']}")
