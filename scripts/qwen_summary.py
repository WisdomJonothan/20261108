#!/usr/bin/env python3
"""Summarise the Qwen3-Reranker-0.6B baseline against our system.

Reads the per-domain score JSONs produced by `ops/score_qwen3.sh` and the
aggregate files for our own arms, and writes `results/QWEN3_BASELINE.md`.

Both systems see the same candidate lists (dense top-100 per query); the
cross-encoder comparison is therefore like-for-like, and the fused comparison
uses the same z-normalised 1:1 weights for both.
"""
import glob
import json
import os

R = "results/"


def per_domain(kind):
    out = {}
    for f in glob.glob(R + "score_%s_*_dev.json" % kind):
        d = os.path.basename(f)[len("score_%s_" % kind):-len("_dev.json")]
        out[d] = json.load(open(f, encoding="utf-8"))
    return out


def macro(rows, key="nDCG@10"):
    return sum(v[key] for v in rows.values()) / len(rows)


def main():
    ce = per_domain("ce_qwen3")
    sub = per_domain("sub_qwen3")
    ours = {
        "lx_possum2": json.load(open(R + "aggregate_sub_lx_possum2_dev.json", encoding="utf-8")),
        "cx_qv": json.load(open(R + "aggregate_sub_cx_qv_dev.json", encoding="utf-8")),
        "sft_e2bc": json.load(open(R + "aggregate_sub_sft_e2bc_dev.json", encoding="utf-8")),
    }
    probe = json.load(open(R + "CE_FUSION_PROBE_FIXED.json", encoding="utf-8"))["models"]
    common = sorted(set(ce) & set(sub))
    lines = ["# Qwen3-Reranker-0.6B baseline (dev)", "",
             "Protocol: official model card (`prefix + budgeted body + think suffix`, "
             "P(yes) from `[no, yes]`), `max_len=2048`, `doc_cap=4000` chars, batch 4-16.",
             "Candidates: the same dense top-100 per query as our reranker.", "",
             "| domain | Qwen CE alone | Qwen + dense (1:1) | our system (frozen recipe) |",
             "|---|---|---|---|"]
    for d in common:
        ours_d = {x["domain"]: x["ndcg10"] for x in ours["lx_possum2"]["domains"]}[d]
        lines.append("| %s | %.4f | %.4f | %.4f |" % (
            d, ce[d]["nDCG@10"], sub[d]["nDCG@10"], ours_d))
    lines.append("| **macro** | **%.4f** | **%.4f** | **%.4f** |" % (
        macro(ce), macro(sub), ours["lx_possum2"]["domain_macro"]))
    lines.append("")
    lines.append("Our cross-encoder alone (same probe, same candidates): **%.4f**." % probe["lx_possum2"]["ce_only"]["macro"])
    txt = "\n".join(lines) + "\n"
    open(R + "QWEN3_BASELINE.md", "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
