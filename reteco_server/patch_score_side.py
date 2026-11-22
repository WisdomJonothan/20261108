"""Record the 2026-11-22 evening score-side experiments in REPORT, README, paper, claims."""
import json
from pathlib import Path

R = Path("results/replacement_v2")
BS = chr(92)

# 1. REPORT.md
rep = R / "REPORT.md"
s = rep.read_text(encoding="utf-8")
block = """
## 11. Score-side experiments (2026-11-22 evening)

Three experiments asked whether the *first stage* rather than the judge is the bottleneck.

**The frozen system is reproducible end-to-end.** Rerunning the frozen checkpoint
(`results/ce_ft/v2m3_pool8_e2/best`) through the documented chain (dense top-100, title
query capped at 256 tokens, window 250/192/3, pair-max-len 510, z 1:1 fusion) gives
**0.3581821** against the recorded **0.3580985** (delta 8.4e-5); 11 of 13 domains agree to
four decimals, economics and workplace differ by 0.0006 and 0.0017. The original CE run
files are not in the repository, which is why the earlier `fuse_scores.py` probe on
`runs/fusion100_full_ce_s17_*` could not match the submission; with the frozen checkpoint
rerun the pipeline reproduces.

**A slate-metric loss does not help the cross-encoder.** Retraining the same CE with
`--loss ndcg` (lambdarank with delta-nDCG@10 pair weights, 969 train queries, 2 epochs,
lr 1.5e-5, seed 17) and evaluating through the identical chain gives **0.3519337**, i.e.
**-0.0062** against the reproduction. Five domains improve (travel +0.0074, cardano +0.0026,
iota +0.0028, bitcoin +0.0013, quant +0.0009) and eight lose, worst monero -0.0215,
history -0.0195, hsm -0.0181, law -0.0149. The loss function is not the limiting factor at
this data scale.

**The 1:1 fusion weight cannot be improved from train.** Sweeping w in {0.5, 0.75, 1, 1.25,
1.5, 2, 2.5, 3} on the 969 train queries with the leak-free out-of-fold CE runs selects
**w = 0.75** (train macro 0.2811 vs 0.2789 at w = 1), but on dev that choice gives **0.3486**
while w = 1 gives 0.3582. The train-optimal weight transfers negatively, so the frozen 1:1
choice is kept. The dev-optimal w = 1.5 (0.3591) was observed after the fact and is not
adopted.

Together these rule out "the CE was trained for the wrong objective" and "the fusion weight
was never tuned" as explanations of the remaining gap; the gap sits in the cross-encoder's
ranking quality itself (pool oracle 0.7446 versus 0.3581 delivered).
"""
s = s.rstrip() + "\n" + block
rep.write_text(s, encoding="utf-8")

# 2. README.md bullet
readme = Path("README.md")
s = readme.read_text(encoding="utf-8")
anchor = "## Replacement study (dev, 2026-11-20)"
bullet = """**Score-side experiments (2026-11-22).** The frozen system reproduces end-to-end from the
frozen checkpoint through the documented chain (0.3581821 vs recorded 0.3580985, 11/13
domains equal to four decimals). A lambdarank nDCG@10 retraining of the same CE scores
0.3519 (-0.0062), and a fusion-weight sweep selected on train (w=0.75) drops dev to 0.3486
against 0.3582 at the frozen w=1, so neither the loss function nor the fusion weight
explains the remaining gap.

"""
if anchor in s and "Score-side experiments (2026-11-22)" not in s:
    s = s.replace(anchor, bullet + anchor, 1)
    readme.write_text(s, encoding="utf-8")

# 3. paper paragraph
paper = Path("paper/main.tex")
s = paper.read_text(encoding="utf-8")
para = (BS + "paragraph{Slate-metric training and the fusion weight.} ")
para += ("Two first-stage explanations for the remaining gap were tested and rejected. Retraining "
         "the cross-encoder with a lambdarank $\Delta$nDCG@10 objective, evaluated through the "
         "identical chain, gives 0.3519 against the reproduction's 0.3582 (five domains up, eight "
         "down), so the loss function is not the binding constraint at 969 training queries. A "
         "sweep of the dense-to-CE fusion weight selects $w=0.75$ on the 969 training queries but "
         "that choice scores 0.3486 on dev against 0.3582 at the frozen $w=1$; the train-optimal "
         "weight does not transfer, and the frozen value is kept.\n\n")
anchor = BS + "paragraph{Transfer to FollowIR.}"
if "Slate-metric training and the fusion weight" not in s and anchor in s:
    s = s.replace(anchor, para + anchor, 1)
    paper.write_text(s, encoding="utf-8")

# 4. claims
cl = Path("scripts/check_claims.py")
s = cl.read_text(encoding="utf-8")
add = '''
    {"label": "frozen checkpoint reproduction (dev domain-macro)",
     "path": "replacement_v2/aggregate_frozenrepro_dev.json", "key": "domain_macro",
     "expect": 0.3581821, "tol": 1e-6, "in_paper": False},
    {"label": "lambdarank nDCG@10 CE (dev domain-macro)",
     "path": "replacement_v2/aggregate_ndcgdev_dev.json", "key": "domain_macro",
     "expect": 0.3519337, "tol": 1e-6, "in_paper": False},
'''
if "aggregate_frozenrepro_dev.json" not in s:
    idx = s.find("]\n")
    s = s[:idx] + add.rstrip("\n") + "\n" + s[idx:]
    cl.write_text(s, encoding="utf-8")

print("patched")
