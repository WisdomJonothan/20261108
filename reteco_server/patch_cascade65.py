"""Record the 65-query pool-100 cascade result and close the specialist-reranker line."""
from pathlib import Path

ROOT = Path(r"D:\RETECO\github_repo")
REPORT = ROOT / "results" / "replacement_v2" / "REPORT.md"
README = ROOT / "README.md"

BLOCK = """
### 16.2b The pool-100 cascade does not hold up either (65 queries)

The 26-query cascade (keep the best five per 20-document group, rerank the 25 survivors) gave
0.185133 -> 0.259044 (+0.074). Extended to 65 queries (5 per domain) the same setting gives
**0.245332 -> 0.232081 (-0.0133)**, zero-score queries 28 -> 25, relevant-in-top-10 55 -> 56.
Both Diver variants are therefore nDCG-neutral or worse once the query sample is not the first
two queries of each domain; what it does do, consistently, is move a handful of additional
relevant documents into the top ten. The specialist-reranker line is closed.
"""


def main() -> None:
    text = REPORT.read_text(encoding="utf-8")
    if "16.2b" not in text:
        text = text.rstrip() + "\n\n" + BLOCK.strip() + "\n"
        REPORT.write_text(text, encoding="utf-8")
        print("REPORT patched")

    marker = "\n\n## 17."
    if "16.2b" not in text:
        text = text.rstrip() + "\n\n" + BLOCK.strip() + "\n"
        REPORT.write_text(text, encoding="utf-8")
        print("REPORT patched")
    elif marker in text:
        text = text.replace(marker, "\n\n" + BLOCK.strip() + marker, 1)
        REPORT.write_text(text, encoding="utf-8")
        print("REPORT patched mid")

    readme = README.read_text(encoding="utf-8")
    if "0.2321" not in readme:
        target = "**Review round 3 (2026-11-24).**"
        idx = readme.find(target)
        assert idx > 0, "review round 3 bullet missing"
        add = (
            "\n  The pool-100 cascade does not survive more queries either: 0.2453 -> 0.2321 "
            "(-0.0133) at 65 queries, so the specialist reranker is nDCG-neutral or worse at any "
            "sample size we have measured.\n"
        )
        end = readme.find("\n\n", idx)
        readme = readme[:end] + add + readme[end:]
        README.write_text(readme, encoding="utf-8")
        print("README patched")


if __name__ == "__main__":
    main()
