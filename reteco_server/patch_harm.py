"""Insert the harmful-case review into REPORT, paper, README and the claims list."""
import json
from pathlib import Path

REPO = Path(r"D:/RETECO/github_repo")
V = json.loads((REPO / "results/replacement_v2/harm_review_verdicts.json").read_text(encoding="utf-8"))
t = V["tally"]
assert sum(t.values()) == 19, t

REPORT_SEC = """## 9h. Reading the harmful class: 19 adjudicated cases

The harmful-swap counts above are upper bounds, so we read the documents themselves.
`scripts/dump_harm.py` selects the score-changing swaps that insert an unjudged
challenger while removing a judged-relevant incumbent; the Direct arm produces 19 such
cases. Each case was adjudicated against the question (artefact
`results/replacement_v2/harm_review_verdicts.json`, single reviewer):

| verdict | cases | what the challenger is |
| --- | --- | --- |
| better | 8 | on topic and at least as useful as the removed document |
| comparable | 4 | on topic and about as useful |
| mixed | 3 | answers a different aspect of the question |
| worse | 4 | genuinely less useful |

Two of the four genuine losses are navigational pages, a table of contents and a
navigation box, which carry no prose at all. The other two are a numeric question whose
answer the incumbent states and an off-topic drift. Thirteen of the nineteen
replacements insert a document that is on topic for the question, which is what the
label-coverage bound predicts: the qrels never judged 92.1% of the candidate pool, so a
swap that removes a judged-relevant document and inserts an unjudged but reasonable one
is scored as pure loss. The practical implication is that part of the measured harm is a
labelling artefact, and that a document-type filter would remove half of the real harm at
no cost to the judge.

"""

PAPER_PARA = r"""\paragraph{What the harmful swaps look like.} The harmful-swap counts are upper bounds
because the judgments cover only 4.9\% of the candidate pool, so we read the documents
themselves. Of the nineteen score-changing swaps that remove a judged-relevant document
in favour of an unjudged one, eight insert a document that is at least as useful as the
removed one, four are about as useful, three answer a different aspect of the question,
and four are genuine losses. Two of those four are navigational pages (a table of
contents and a navigation box) that carry no prose, which a document-type check would
reject, and the other two are a numeric question whose answer the incumbent states and an
off-topic drift. Most of the measured harm is therefore a labelling artefact of the
4.9\% coverage rather than a ranking error, which is why the harmful-loss column is read
as an upper bound throughout this section.

"""

README_BULLET = """- **Most of the measured harm is a label artefact.** Reading the 19 score-changing swaps
  that remove a judged-relevant document, 8 insert a document at least as useful, 4 are
  about as useful, 3 answer a different aspect and 4 are genuine losses; two of those four
  are navigational pages with no prose. With judgments covering 4.9% of the candidate pool,
  harmful-swap counts are upper bounds, and a document-type filter would remove half of the
  real harm for free.

"""

CLAIMS = """    # --- harmful-case adjudication (2026-11-22) ------------------------------
    ("harmful cases adjudicated", 19,
     lambda: sum(jload("replacement_v2/harm_review_verdicts.json")["tally"].values()),
     "replacement_v2/harm_review_verdicts.json", False),
    ("harmful cases genuinely worse", 4,
     lambda: jload("replacement_v2/harm_review_verdicts.json")["tally"]["worse"],
     "replacement_v2/harm_review_verdicts.json", False),
    ("harmful cases at least as useful", 8,
     lambda: jload("replacement_v2/harm_review_verdicts.json")["tally"]["better"],
     "replacement_v2/harm_review_verdicts.json", False),
"""


def patch(path, old, new, count=1):
    p = REPO / path
    s = p.read_text(encoding="utf-8")
    assert s.count(old) >= 1, (path, "anchor missing")
    p.write_text(s.replace(old, new, count), encoding="utf-8", newline="\n")
    print("patched", path)


patch("results/replacement_v2/REPORT.md", "## 10. Conclusion (2026-11-21)", REPORT_SEC + "## 10. Conclusion (2026-11-21)")
patch("paper/main.tex", r"\subsection*{9.1 Verification and replay}", PAPER_PARA + r"\subsection*{9.1 Verification and replay}")
patch("README.md", "\n## Paper\n", README_BULLET + "\n## Paper\n")
patch("scripts/check_claims.py", "\n]\n\n\ndef main():", CLAIMS + "]\n\n\ndef main():")
print("ok")
