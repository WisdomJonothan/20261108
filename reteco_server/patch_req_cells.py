"""Record the four-cell requirement experiment (2026-11-24) in REPORT/README/claims."""
from pathlib import Path

ROOT = Path(r"D:\RETECO\github_repo")
REPORT = ROOT / "results" / "replacement_v2" / "REPORT.md"
README = ROOT / "README.md"

BLOCK = """## 17. Does requirement-conditioned evidence help? (2026-11-24)

Question, fixed in advance: on the same six pairs, the same 4B judge and the same
total input budget, does pointing the judge at the evidence the question's own
requirements ask for make it accept more of the swaps that actually help?

Design. Four cells over the three held-out folds (754 queries that carry at least
two decomposition steps), 12 prompts per query in every cell:

| cell | passage text shown | comparison question |
|---|---|---|
| A (reference) | the 112-word query-overlap window | original relevance question |
| sel | 112-word window chosen to cover the requirement | original relevance question |
| req | the original window | which passage better provides what the requirement asks for |
| both | requirement-selected window | requirement-conditioned question |
| mis | window selected for a *sibling* requirement | question asks the sibling requirement |

`mis` is the decisive control: same machinery, same length, wrong requirement. If
the gain is about the requirement matching the evidence, `req` must beat `mis`.

Out-of-fold results (frozen baseline 0.300255 on these queries):

| cell | nDCG@10 | vs frozen | vs cell A | changed / helped / hurt |
|---|---:|---:|---:|---|
| A six pairs | **0.307869** | +0.007614 | — | — |
| sel | 0.304220 | +0.003965 [-0.000638, +0.008746] | **-0.003649 [-0.007432, -0.000266]** | 41 / 18 / 23 |
| req | 0.306841 | +0.006586 [+0.001801, +0.011832] | -0.001028 [-0.005928, +0.004083] | 43 / 18 / 25 |
| both | 0.304290 | +0.004035 [-0.000338, +0.008824] | -0.003580 [-0.008062, +0.000903] | 58 / 23 / 35 |
| mis | 0.305816 | +0.005561 [+0.001616, +0.010270] | -0.002054 [-0.006330, +0.002501] | 61 / 26 / 35 |

Matched minus mismatched requirement: **+0.001025, 95% CI [-0.003317, +0.005464],
p = 0.327**. Selection on the requirement alone is *worse* than the frozen rule by
0.0036 with an interval that excludes zero; adding the requirement question does
not recover it; the requirement question on its own is neutral.

Reading. The experiment was designed so that a positive result would have to show
up as `req` > `mis`. It does not. Whatever the cells buy comes from changing what
the judge is asked, not from the requirement being the right one, and the version
that changes the passages instead of the question loses ground. This matches the
other three measurements of the same mechanism: step-level support judgement is
worth about one AUC point over the global question (section 16.4), requirement-
bearing text is equally often outside the shown window in successful (38.6%) and
failed (30.2%) swaps, and 55% of the missed beneficial swaps are confident
refusals below p = 0.35 rather than agreement-rule vetoes (26%).

Consequence for the competition configuration: nothing here beats cell A, so the
frozen 0.3735 configuration is unchanged. Per the pre-agreed stopping rule (a
candidate must beat the six-pair rule out of fold and beat the mismatched
requirement), this line stops before any dev confirmation, before expanding six
pairs to eighteen and before allowing a second swap.
"""


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    if "## 17. Does requirement-conditioned evidence help" not in text:
        REPORT.write_text(text.rstrip() + "\n\n" + BLOCK, encoding="utf-8")
        print("REPORT patched")
    readme = README.read_text(encoding="utf-8")
    if "requirement-conditioned evidence" not in readme:
        lines = readme.split("\n")
        for i, ln in enumerate(lines):
            if ln.startswith("- **Review round 3 (2026-11-24):**"):
                lines.insert(i + 1, (
                    "  **Requirement-conditioned evidence (2026-11-24):** on the held-out folds "
                    "(754 queries) neither selecting passages by the question's requirement "
                    "(-0.0036 vs the six-pair rule, CI [-0.0074,-0.0003]) nor asking the judge the "
                    "requirement-conditioned question (-0.0010, ns) helps, and the matched "
                    "requirement beats a mismatched one by only +0.0010 (CI [-0.0033,+0.0055]). "
                    "The six-pair configuration stays as the frozen competition entry."))
                break
        else:
            raise SystemExit("README anchor missing")
        README.write_text("\n".join(lines), encoding="utf-8")
        print("README patched")


if __name__ == "__main__":
    main()
