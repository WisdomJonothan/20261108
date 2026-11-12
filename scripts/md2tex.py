#!/usr/bin/env python3
"""Convert docs/PAPER_4PAGE.md into an ACL-style LaTeX document (paper/main.tex).

Supported Markdown subset: `#`/`##`/`###` headings, paragraphs, pipe tables,
`-` bullets, `**bold**`, `*italic*`, `` `code` ``, and the Unicode symbols that
appear in the text.  The References section is converted to a real
`thebibliography` so `\\cite{...}` works.
"""
import re
import sys

PRE = r"""\documentclass[11pt]{article}
\PassOptionsToPackage{numbers}{natbib}
\usepackage[review]{acl}
\makeatletter\acl@anonymizefalse\makeatother
\usepackage{times}
\usepackage{latexsym}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage[htt]{hyphenat}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{graphicx}
\sloppy
\setlength{\tabcolsep}{4pt}
\title{%(title)s}
\author{Author Name \\ Affiliation \\ \texttt{email@example.org}}
\date{}
\begin{document}
\maketitle
"""

UNI = {
    "\u0394": r"$\Delta$", "\u03b1": r"$\alpha$", "\u03b2": r"$\beta$",
    "\u03b3": r"$\gamma$", "\u03b4": r"$\delta$", "\u03b5": r"$\epsilon$",
    "\u03bb": r"$\lambda$", "\u03bc": r"$\mu$", "\u03c3": r"$\sigma$",
    "\u03c4": r"$\tau$", "\u03b8": r"$\theta$", "\u2248": r"$\approx$",
    "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u00b1": r"$\pm$",
    "\u00d7": r"$\times$", "\u2192": r"$\to$", "\u2212": "-",
    "\u2014": "---", "\u2013": "--", "\u2026": r"\ldots{}",
    "\u2019": "'", "\u2018": "`", "\u201c": "``", "\u201d": "''",
    "\u00b0": r"$^{\circ}$", "\u00a0": " ", "\u00b7": r"$\cdot$",
}


def esc(s):
    out = []
    for ch in s:
        if ch in UNI:
            out.append(UNI[ch])
        elif ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "'":
            out.append("\u2019")
        else:
            out.append(ch)
    s = "".join(out)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = re.sub(r"`(.+?)`", r"\\texttt{\1}", s)
    def _cite(m):
        keys = re.findall(r"@([A-Za-z0-9_]+)", m.group(1))
        return r"\citep{" + ",".join(keys) + "}"
    s = re.sub(r"\[(@[A-Za-z0-9_]+(?:;\s*@[A-Za-z0-9_]+)*)\]", _cite, s)
    return s


def table(rows):
    header, body = rows[0], rows[1:]
    cols = len(header)
    spec = "l" + "r" * (cols - 1)
    wide = cols >= 6
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    size = r"\scriptsize" if wide else r"\footnotesize"
    lines = [r"\begin{%s}[t]" % env, r"\centering", size,
             r"\resizebox{%s}{!}{%%" % width,
             r"\begin{tabular}{%s}" % spec, r"\toprule",
             " & ".join(esc(c.strip()) for c in header) + r" \\", r"\midrule"]
    for r in body:
        r = r + [""] * (cols - len(r))
        lines.append(" & ".join(esc(c.strip()) for c in r[:cols]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}%", r"\end{%s}" % env]
    return "\n".join(lines)

def convert(md):
    lines = md.splitlines()
    title = "Multi-evidence re-ranking for RETECO Track 1a"
    body = []
    para = []
    bullets = []
    state = {"abstract": False, "refs": False}

    def flush_para():
        nonlocal para
        if not para:
            return
        text = " ".join(para)
        para = []
        if state["refs"]:
            m = re.match(r"^\[([A-Za-z0-9_]+)\]\s*(.+)$", text.strip(), re.S)
            if m:
                clean = re.sub(r"\(`%s`\)$" % re.escape(m.group(1)), "",
                               m.group(2)).strip()
                yr = re.search(r"\b(19\d\d|20\d\d)\b", clean)
                yr = yr.group(1) if yr else "n.d."
                first = re.split(r",| and ", clean)[0].strip()
                surname = first.split()[-1] if first.split() else "Anon"
                many = "," in clean.split(str(yr))[0]
                label = "%s%s(%s)" % (surname, " et al." if many else "", yr)
                body.append(r"\bibitem[%s]{%s} %s"
                            % (esc(label), m.group(1), esc(clean)))
            return
        body.append(esc(text))
        body.append("")

    def flush_bullets():
        nonlocal bullets
        if bullets:
            body.append(r"\begin{itemize}")
            for b in bullets:
                body.append(r"\item " + esc(b))
            body.append(r"\end{itemize}")

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("# ") and not s.startswith("## "):
            title = s[2:].split("(")[0].strip()
            i += 1
            continue
        if s.startswith("## "):
            flush_para(); flush_bullets()
            name = s[3:].strip()
            if state["abstract"]:
                body.append(r"\end{abstract}")
                state["abstract"] = False
            low = name.lower()
            if low == "abstract":
                body.append(r"\begin{abstract}")
                state["abstract"] = True
            elif low.startswith("references"):
                state["refs"] = True
                body.append(r"\begin{thebibliography}{99}")
                body.append(r"\small")
            else:
                body.append(r"\section*{%s}" % esc(name))
            i += 1
            continue
        if s.startswith("### "):
            flush_para(); flush_bullets()
            body.append(r"\subsection*{%s}" % esc(s[4:].strip()))
            i += 1
            continue
        if s.startswith(">"):
            i += 1
            continue
        if s.startswith("|"):
            flush_para(); flush_bullets()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.match(r"^[\s:\-|]+$", lines[i].strip()):
                    rows.append(lines[i].strip().strip("|").split("|"))
                i += 1
            body.append(table(rows))
            body.append("")
            continue
        if s.startswith("- "):
            flush_para()
            bullets.append(s[2:].strip())
            i += 1
            continue
        if not s:
            flush_para(); flush_bullets()
            i += 1
            continue
        para.append(s)
        i += 1
    flush_para(); flush_bullets()
    if state["abstract"]:
        body.append(r"\end{abstract}")
    if state["refs"]:
        body.append(r"\end{thebibliography}")
    return PRE % {"title": esc(title)} + "\n".join(body) + "\n\\end{document}\n"


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "docs/PAPER_4PAGE.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else "paper/main.tex"
    tex = convert(open(src, encoding="utf-8").read())
    open(dst, "w", encoding="utf-8").write(tex)
    print("wrote", dst, len(tex), "chars")
