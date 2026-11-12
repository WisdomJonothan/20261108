"""Check the numbers printed in the paper against the artefacts that produced them.

Every claim is (label, paper value, extractor, source path). The extractor pulls a
number out of a committed JSON/Markdown artefact; the check passes when it rounds
to the paper's 4-decimal value (or is within 2e-4, for aggregations that differ by
a rounding step). Exits non-zero if any claim fails.
"""
import json
import sys

R = "results/"


def jload(p):
    with open(R + p, encoding="utf-8") as f:
        return json.load(f)


def dom_macro(p):
    return jload(p)["domain_macro"]


def boot(p, key):
    return jload(p)[key]["mean"], jload(p)[key]["ci_lo"], jload(p)[key]["ci_hi"]


def probe(model, arm):
    return jload("CE_FUSION_PROBE.json")["models"][model][arm]["macro"]


def md_number(p, needle):
    txt = open(R + p, encoding="utf-8").read()
    i = txt.index(needle)
    return float(txt[i:i + 200].split("|")[1].strip())


CLAIMS = [
    ("frozen dev nDCG@10", 0.3581, lambda: dom_macro("aggregate_sub_sft_e2bc_dev.json"),
     "aggregate_sub_sft_e2bc_dev.json"),
    ("frozen gain vs zero-shot", 0.0332,
     lambda: boot("boot_dom_e2_vs_zs.json", "domain_bootstrap")[0], "boot_dom_e2_vs_zs.json"),
    ("frozen CI low", 0.0192,
     lambda: boot("boot_dom_e2_vs_zs.json", "domain_bootstrap")[1], "boot_dom_e2_vs_zs.json"),
    ("frozen CI high", 0.0472,
     lambda: boot("boot_dom_e2_vs_zs.json", "domain_bootstrap")[2], "boot_dom_e2_vs_zs.json"),
    ("dense dev nDCG@10", 0.2943, lambda: dom_macro("aggregate_dense_mxbai_dev.json"),
     "aggregate_dense_mxbai_dev.json"),
    ("zero-shot CE dev", 0.3249, lambda: dom_macro("aggregate_sub_zs11_dev.json"),
     "aggregate_sub_zs11_dev.json"),
    ("full-question arm dev", 0.3443, lambda: dom_macro("aggregate_sub_cx_qv_dev.json"),
     "aggregate_sub_cx_qv_dev.json"),
    ("same-code retrain dev", 0.3470, lambda: dom_macro("aggregate_sub_lx_possum_dev.json"),
     "aggregate_sub_lx_possum_dev.json"),
    ("deployment-budget epoch dev", 0.3428, lambda: dom_macro("aggregate_sub_lx_ep1_dev.json"),
     "aggregate_sub_lx_ep1_dev.json"),
    ("cx_qv fused at best w", 0.3546, lambda: probe("cx_qv", "w=2"), "CE_FUSION_PROBE.json"),
    ("cx_qv cross-encoder alone", 0.3153, lambda: probe("cx_qv", "ce_only"), "CE_FUSION_PROBE.json"),
    ("lx_possum cross-encoder alone", 0.2756, lambda: probe("lx_possum", "ce_only"),
     "CE_FUSION_PROBE.json"),
    ("cx_qv fusion at w=1", 0.3443, lambda: probe("cx_qv", "w=1"), "CE_FUSION_PROBE.json"),
    # Not our computation: the organisers' published Track 1a dev baseline.
    ("BM25 baseline (organisers)", 0.1147,
     lambda: float(open(R + "PHASE9_PLAN.md", encoding="utf-8").read()
                   .split("official BM25 1a dev baseline (organizers)")[1]
                   .split("|")[1].strip()),
     "PHASE9_PLAN.md (organisers' data page)"),
]


def main():
    ok = True
    rows = []
    for label, expected, fn, src in CLAIMS:
        try:
            got = float(fn())
        except Exception as e:  # noqa: BLE001
            rows.append((label, expected, None, src, "ERROR %s" % e))
            ok = False
            continue
        close = abs(round(got, 4) - expected) <= 2e-4
        rows.append((label, expected, got, src, "ok" if close else "MISMATCH"))
        ok = ok and close
    w = max(len(r[0]) for r in rows)
    for label, expected, got, src, status in rows:
        g = "n/a" if got is None else "%.4f" % got
        print("%-*s paper %.4f  artefact %s  %-9s %s" % (w, label, expected, g, status, src))
    print("\nRESULT:", "all claims traceable" if ok else "CLAIM MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
