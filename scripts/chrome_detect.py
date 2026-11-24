"""Document-only boilerplate / navigation / page-chrome detector.

Deployable counterpart of the 4B-model DUP probe (``temporal_audit.py``):
same target ("is this document mostly navigation, boilerplate, a heading-only
fragment or a table-of-contents block?"), but scored from the document text
alone -- no query, no qrels, no LLM, no GPU.

Design
------
17 bounded features in [0,1] (see ``FEATURES``), each a *rate* per 100 words
(or a fraction of lines / characters) divided by a reference scale where the
value saturates at 1.0.  The score is a fixed-weight logistic:

    z = b + sum_i w_i * f_i
    chrome = 1 / (1 + exp(-z))

Weights are hand-set from the audit's error taxonomy (navigation words > link
and markup density > layout tokens > length), *not* fitted to the model
labels, so the AUC reported against ``p["DUP"] > 0.5`` is an honest
out-of-sample measure.  The weights are deliberately round numbers so the
score can be read off by hand: z = -4.2 means "no chrome evidence at all",
z = +2.2 (score 0.90) needs strong evidence from several families.

Usage
-----
    from chrome_detect import score_text, extract_features
    score_text("Share this on Twitter | Home > Blog | Copyright 2023")

    python chrome_detect.py --documents data/track1_tempo/bitcoin/documents.jsonl \
        --limit 1000 --out /tmp/chrome_sample.jsonl
"""
import argparse
import json
import math
import re
import sys

# ------------------------------------------------------------------ lexicons
# weight = how diagnostic the phrase is on its own (3 = chrome by itself,
# 2 = strong, 1 = weak / ambiguous).  Matching is case-insensitive substring.
NAV_PHRASES = {
    # social / sharing chrome
    "share this": 3, "share on": 2, "tweet": 2, "facebook": 1, "twitter": 1,
    "linkedin": 1, "reddit": 1, "pinterest": 1, "whatsapp": 1, "email this": 2,
    "print this": 2, "add to favorites": 2,
    # account / cookie / legal chrome
    "sign in": 2, "log in": 2, "login": 2, "sign up": 2, "sign-up": 2,
    "register": 1, "subscribe": 2, "newsletter": 2, "cookie": 2,
    "privacy policy": 2, "terms of service": 2, "terms and conditions": 2,
    "all rights reserved": 3, "copyright": 2, "©": 2, "consent": 1,
    "we use cookies": 3, "accept all": 2,
    # wiki / CMS chrome
    "jump to": 2, "skip to": 2, "suggest edit": 3, "suggest an edit": 3,
    "edit this page": 3, "edit links": 2, "cite this page": 3,
    "printable version": 3, "permanent link": 2, "wikidata": 2,
    "from wikipedia": 2, "contributors": 2, "contributor": 1,
    "table of contents": 3, "contents": 1, "see also": 1, "external links": 2,
    "references": 1, "navigation": 1, "menu": 1, "breadcrumb": 2,
    "home >": 3, "you are here": 3, "archived from": 2, "retrieved": 1,
    "last edited": 2, "categories:": 1, "related articles": 1,
    "advertisement": 2, "sponsored": 2, "read more": 2, "click here": 3,
    "learn more": 1, "continue reading": 1, "older posts": 2, "newer posts": 2,
    "previous": 1, "next": 1, "page 1 of": 2, "showing": 1,
    "search": 1, "toggle": 1, "dropdown": 1, "close": 1,
}
# phrase families for the composite "chrome phrase" feature (spec-level hits)
CHROME_STRONG = ["share this", "share on", "suggest edit", "suggest an edit",
                 "table of contents", "jump to", "skip to", "cite this page",
                 "printable version", "click here", "home >", "you are here",
                 "all rights reserved", "we use cookies", "email this",
                 "print this", "add to favorites", "permanent link",
                 "edit this page", "edit links"]

URL_RE = re.compile(r"https?://|www\.", re.I)
MD_LINK_RE = re.compile(r"\[[^\]\n]{0,120}\]\(")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,12}[ >/]")
BULLET_RE = re.compile(r"^[\-\*•·▪◦‣–]\s|^\d{1,2}[.)]\s")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
NONPROSE_RE = re.compile(r"[^A-Za-z0-9\s.,;:!?'\"()\-/%$&+’‘“”]")
SPECIAL_RE = re.compile(r"[|•·▪◦‣»«…\[\]{}<>*=#~^_]")
TERMINAL_RE = re.compile(r"[.!?:;\"')\]]\s*$")


def _clip(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def _rate(hits, n_words, ref, cap=1.0):
    """hits per 100 words, saturating at ``ref``."""
    if n_words <= 0:
        return 0.0
    return min(cap, (hits * 100.0 / n_words) / ref)


def extract_features(text):
    """Return the 14 bounded features of ``text`` as a dict (all in [0,1])."""
    text = text or ""
    n_chars = max(len(text), 1)
    words = WORD_RE.findall(text)
    nw = len(words)
    n_words = max(nw, 1)
    low = text.lower()
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    n_lines = max(len(lines), 1)

    # ---- length -------------------------------------------------------
    f_short = 1.0 if nw < 40 else (0.6 if nw < 70 else (0.3 if nw < 110 else 0.0))

    # ---- character composition ---------------------------------------
    nonprose = len(NONPROSE_RE.findall(text)) / n_chars
    special = len(SPECIAL_RE.findall(text)) / n_chars
    digits = sum(ch.isdigit() for ch in text) / n_chars
    alpha = [ch for ch in text if ch.isalpha()]
    upper = (sum(ch.isupper() for ch in alpha) / max(len(alpha), 1))
    f_nonprose = _clip(nonprose / 0.08)
    f_special = _clip(special / 0.05)
    f_upper = _clip((upper - 0.08) / 0.30)
    f_digit = _clip(digits / 0.12)

    # ---- link / markup density ---------------------------------------
    urls = len(URL_RE.findall(text))
    md_links = len(MD_LINK_RE.findall(text))
    tags = len(HTML_TAG_RE.findall(text))
    f_links = _rate(urls + md_links, nw, 2.5)
    f_html = _rate(tags, nw, 2.0)

    # ---- navigation lexicon ------------------------------------------
    nav_hits = 0.0
    strong_hits = 0
    for ph, wgt in NAV_PHRASES.items():
        c = low.count(ph)
        if c:
            nav_hits += wgt * min(c, 3)
            if ph in CHROME_STRONG:
                strong_hits += min(c, 2)
    f_nav = _rate(nav_hits, nw, 6.0)
    f_chrome = _rate(strong_hits, nw, 1.2)

    # footer: legal / copyright block in the last 300 chars
    tail = low[-300:]
    f_footer = 1.0 if ("©" in tail or "all rights reserved" in tail
                       or "copyright" in tail or "privacy" in tail) else 0.0

    # ---- layout: heading-only lines, pipes, bullets -------------------
    head_like = 0
    for ln in lines:
        if len(ln) <= 60 and not TERMINAL_RE.search(ln) and len(ln.split()) <= 9:
            head_like += 1
    f_headlines = _clip((head_like / n_lines - 0.15) / 0.55)
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    pipes = text.count("|")
    f_pipes = _clip((max(pipes / n_lines, pipe_lines / n_lines) - 0.2) / 1.8)
    bullets = sum(1 for ln in lines if BULLET_RE.match(ln))
    f_bullets = _clip((bullets / n_lines - 0.15) / 0.55)

    # ---- fragment / quote / self-reference chrome ---------------------
    # mid-sentence truncation is the dominant "fragment" cue in RETECO:
    # passage dumps cut at a word boundary and the tail carries no period.
    tail_stripped = text.rstrip()
    if tail_stripped:
        last = tail_stripped[-1]
        cut = 1.0 if (last.isalnum() or last in ",;:—–-") else 0.0
        ell = 1.0 if tail_stripped.endswith(("...", "…", "[...]")) else 0.0
        lead = 1.0 if (text[:1].islower() or text[:1].isdigit()) else 0.0
        f_fragment = _clip(max(cut, ell, 0.6 * ell + 0.4 * lead))
    else:
        f_fragment = 1.0
    qlines = sum(1 for ln in lines if ln.startswith(">"))
    f_quote = _clip((qlines / n_lines - 0.1) / 0.5)
    selfref = sum(low.count(p) for p in (
        "this question", "my question", "the question is", "as asked",
        "as the title says", "i am not sure", "any help", "thanks in advance",
        "see above", "below is", "edit:", "update:"))
    f_selfref = _rate(selfref, nw, 2.0)

    # ---- repetitiveness ----------------------------------------------
    toks = [wd.lower() for wd in words[:200]]
    if toks:
        from collections import Counter
        cc = Counter(toks).most_common(5)
        top5 = sum(c for _, c in cc) / len(toks)
    else:
        top5 = 0.0
    f_repeat = _clip((top5 - 0.22) / 0.30)

    return {
        "short": f_short,
        "nonprose": f_nonprose,
        "special": f_special,
        "upper": f_upper,
        "digit": f_digit,
        "links": f_links,
        "html": f_html,
        "nav": f_nav,
        "chrome_phrase": f_chrome,
        "footer": f_footer,
        "headlines": f_headlines,
        "pipes": f_pipes,
        "bullets": f_bullets,
        "repeat": f_repeat,
        "fragment": f_fragment,
        "quote": f_quote,
        "selfref": f_selfref,
    }


FEATURES = ["nav", "chrome_phrase", "links", "html", "pipes", "headlines",
            "bullets", "footer", "nonprose", "special", "repeat", "short",
            "fragment", "upper", "digit", "quote", "selfref"]

WEIGHTS = {
    "nav": 2.6,            # navigation / social / legal words -- strongest
    "chrome_phrase": 2.2,  # spec-level phrases ("suggest edit", "home >")
    "links": 1.4,          # raw URL / markdown-anchor density
    "html": 1.2,           # tag density (unrendered chrome)
    "pipes": 1.3,          # "a | b | c" navigation bars
    "headlines": 1.0,      # heading-only lines
    "bullets": 0.8,
    "footer": 1.2,         # copyright / privacy block at the end
    "nonprose": 0.8,
    "special": 0.7,
    "repeat": 0.6,
    "short": 1.6,          # < 40 words is a fragment
    "fragment": 1.1,       # cut mid-sentence / no terminal punctuation
    "quote": 0.6,          # "> " mail-quote blocks
    "selfref": 0.7,        # "this question", "thanks in advance" forum chrome
    "upper": 0.4,
    "digit": 0.3,
}
BIAS = -4.2
TEMP = 1.0


def score_features(f):
    z = BIAS / TEMP
    for k, wgt in WEIGHTS.items():
        z += (wgt / TEMP) * f[k]
    return 1.0 / (1.0 + math.exp(-z))


def score_text(text):
    """Chrome probability in [0,1] for a raw document string."""
    return score_features(extract_features(text))


def score_record(rec):
    """Chrome probability for a corpus record ``{"id":..., "content":...}``."""
    return score_text(rec.get("content") or "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--documents", required=True,
                    help="jsonl file or directory of jsonl files")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="-")
    ap.add_argument("--full", action="store_true", help="dump all features")
    args = ap.parse_args(argv)

    import os
    files = ([os.path.join(args.documents, f)
              for f in sorted(os.listdir(args.documents))
              if f.endswith(".jsonl")]
             if os.path.isdir(args.documents) else [args.documents])
    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    n = 0
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                f = extract_features(rec.get("content") or "")
                row = {"id": rec.get("id"), "chrome": round(score_features(f), 4)}
                if args.full:
                    row["features"] = {k: round(v, 4) for k, v in f.items()}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if args.limit and n >= args.limit:
                    break
        if args.limit and n >= args.limit:
            break
    if out is not sys.stdout:
        out.close()
    print(f"[chrome_detect] scored {n} documents", file=sys.stderr)


if __name__ == "__main__":
    main()
