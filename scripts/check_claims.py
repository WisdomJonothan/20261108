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
    ("BM25 baseline (organisers)", 0.1147,
     lambda: float(open(R + "PHASE9_PLAN.md", encoding="utf-8").read()
                   .split("official BM25 1a dev baseline (organizers)")[1]
                   .split("|")[1].strip()),
     "PHASE9_PLAN.md (organisers' data page)", True),
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
