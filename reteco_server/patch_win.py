"""Insert the evidence-selection result (114-word budget) into REPORT/paper/README/claims."""
from pathlib import Path

REPO = Path(r"D:/RETECO/github_repo")

REPORT_SEC = """## 9i. Choosing the 112 words: same budget, different windows

The 400-word arm is better than the 112-word arm, so the next question is whether the
gain comes from more text or from better-chosen text. We kept the candidate pool, the
nine pairs, the prompt, the model, the 112-word display budget and the decision rule
fixed, and changed only which 112 words the judge sees (`scripts/localized_judge.py`):
`cond` takes the window that covers the question's conditions best (content terms,
numbers and dates quoted in the question, capitalised entity words), and `two` spends
the same 56+56 words on the best condition window plus the best non-adjacent one.

| arm | what the judge sees | nDCG@10 | delta | 95% CI | helpful | harmful | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct | one 112-word overlap window | 0.3646 | +0.0065 | [+0.0021,+0.0117] | 32 | 19 | 5.31 M |
| cond | one 112-word condition window | 0.3645 | +0.0064 | [+0.0020,+0.0116] | 33 | 17 | 5.32 M |
| two | two 56-word windows | 0.3613 | +0.0032 | [-0.0027,+0.0092] | 33 | 18 | 5.33 M |
| wide400 | one 400-word window | 0.3660 | +0.0079 | [+0.0030,+0.0135] | 36 | 10 | 8.22 M |

Paired against the frozen 112-word rule: `cond` +0.00047 [-0.00064,+0.00157] and `two`
-0.00087 [-0.00381,+0.00198], i.e. no difference either way. Choosing the window by the
question's conditions does not reproduce the 400-word gain, and splitting the same budget
over two windows loses a little context. The conclusion of the evidence axis is therefore
narrow and negative: the judge benefits from more raw text, not from a better-chosen 112
words, and even the 400-word arm's paired interval against the 112-word rule includes
zero (-0.0008, +0.0044). At this point neither a stronger model nor smarter selection
moves the result reproducibly, which is the stop condition of the capability/evidence
programme; the system paper is written around the frozen Direct configuration.

"""

PAPER_PARA = r"""\paragraph{Same budget, different windows.} The 400-word arm beats the 112-word
arm, so we ask whether more text or better-chosen text is responsible. Keeping the pool,
the pairs, the prompt, the model, the 112-word display budget and the decision rule
fixed, we give the judge either the 112-word window that best covers the question's
conditions (content terms, numbers/dates quoted in the question, capitalised entities) or
two 56-word windows (that window plus the best non-adjacent one). The condition window
scores 0.3645 ($+0.0064$, CI $[+0.0020,+0.0116]$) and the two-window arm 0.3613
($+0.0032$) against 0.3646 for the frozen overlap window; paired against it the
intervals are $[+0.0016,-0.0006]$ and $[-0.0038,+0.0020]$, so neither selection rule
moves the score. Combined with the capability row, the only lever that changes the
harmful-swap count is the amount of raw text the judge reads (19 to 10 for 400 words),
and even that paired interval includes zero. We therefore stop the capability/evidence
programme here and report the frozen 112-word configuration, which is also the cheapest.

"""

README_BULLET = """- **Better-chosen 112 words do not recover the 400-word gain.** With the pool, pairs,
  prompt, model and decision rule fixed, a window chosen by the question's conditions gives
  0.3645 (+0.0064) and two 56-word windows 0.3613 (+0.0032), against 0.3646 for the frozen
  overlap window; paired intervals [-0.0006,+0.0016] and [-0.0038,+0.0020]. Only the amount of
  raw text moves the harmful-swap count (19 to 10 at 400 words), and that interval also
  includes zero, so the capability/evidence programme stops here and the frozen 112-word
  configuration is what we report.

"""

CLAIMS = """    # --- evidence selection at a fixed display budget (2026-11-22) ------------
    ("condition window nDCG@10", 0.3645,
     lambda: jload("replacement_v2/exp_cond_win_dev.json")["macro_policy"],
     "replacement_v2/exp_cond_win_dev.json", False),
    ("condition window gain", 0.0064,
     lambda: jload("replacement_v2/exp_cond_win_dev.json")["macro_delta"],
     "replacement_v2/exp_cond_win_dev.json", False),
    ("two-window nDCG@10", 0.3613,
     lambda: jload("replacement_v2/exp_two_win_dev.json")["macro_policy"],
     "replacement_v2/exp_two_win_dev.json", False),
    ("two-window gain", 0.0032,
     lambda: jload("replacement_v2/exp_two_win_dev.json")["macro_delta"],
     "replacement_v2/exp_two_win_dev.json", False),
"""


def patch(path, old, new):
    p = REPO / path
    s = p.read_text(encoding="utf-8")
    assert old in s, (path, "anchor missing")
    p.write_text(s.replace(old, new, 1), encoding="utf-8", newline="\n")
    print("patched", path)


patch("results/replacement_v2/REPORT.md", "## 10. Conclusion (2026-11-21)",
      REPORT_SEC + "## 10. Conclusion (2026-11-21)")
patch("paper/main.tex", r"\subsection*{9.1 Verification and replay}",
      PAPER_PARA + r"\subsection*{9.1 Verification and replay}")
patch("README.md", "\n## Paper\n", README_BULLET + "\n## Paper\n")
patch("scripts/check_claims.py", "\n]\n\n\ndef main():", CLAIMS + "]\n\n\ndef main():")
print("ok")
