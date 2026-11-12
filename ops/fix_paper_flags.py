import io

p = r"D:\RETECO\docs\PAPER_DRAFT.md"
s = io.open(p, encoding="utf-8").read()

subs = [
    # (a) "only" -> "largest": the 1-epoch arm's domain CI also excludes zero
    ("[+0.0192,+0.0472]) — the only single-factor effect in this paper whose\ndomain-clustered interval excludes zero",
     "[+0.0192,+0.0472]) — the largest single-factor effect in this paper, and one\nwhose domain-clustered interval excludes zero"),
    # (b) label the CE-alone row as the 1-epoch model and add the frozen arm's CE
    ("| SFT CE alone | 0.2742 <!-- results/aggregate_sft_ce_dev.json --> | 0.3096 <!-- results/aggregate_sft_ce_dev.json --> |",
     "| SFT CE alone (1 epoch) | 0.2742 <!-- results/aggregate_sft_ce_dev.json --> | 0.3096 <!-- results/aggregate_sft_ce_dev.json --> |\n"
     "| SFT CE alone (2 epochs) | 0.2823 <!-- re-scored runs/rerank_sft_e2bc_*_dev.trec; see results/PAPER_VERIFY_E2.md --> | — |"),
    # (c) the 1-epoch sweep is not flat to +-0.008 when w=1.8 reaches +0.0149
    ("also *flat* to within about ±0.008 with no monotone trend",
     "also *flat* to within about ±0.015 with no monotone trend"),
    # (d) the 2-epoch sweep is wider, not flatter
    ("The same sweep for the 2-epoch arm is flatter and always positive",
     "The same sweep for the 2-epoch arm is always positive (and wider than the\n1-epoch one, so it is not evidence of a flat surface)"),
    # capture numbers now come from the regenerated e2 file
    ("<!-- results/CAPTURE_CURVE.md; results/aggregate_sub_sft_e2bc_dev.json -->",
     "<!-- results/CAPTURE_CURVE_E2.md; results/capture_curve_e2.json -->"),
]

for old, new in subs:
    assert old in s, old[:60]
    s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("paper flags fixed:", len(subs))
