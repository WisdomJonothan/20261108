#!/usr/bin/env python
"""Derive cap-1 and cap-2 candidate lists from the frozen cap-3 list (budget curve).

The frozen protocol caps candidates at 3 and orders them by best rank in
dense/full-CE (ties: dense first, then docid).  A smaller budget is exactly the
prefix of that ordered list, so the files are produced by filtering, not by
re-running the builder; the pair list is filtered to pairs whose challenger is
still present, which keeps slot x candidate structure intact.
"""
import json
from pathlib import Path

R = Path("results/replacement_v2")
src = json.loads((R / "candidates_dev.json").read_text())
for k in (1, 2):
    rows = []
    for row in src["queries"]:
        keep = row["candidates"][:k]
        ids = {c["docid"] for c in keep}
        rows.append({**row, "candidates": keep,
                     "pairs": [p for p in row["pairs"] if p["incoming_docid"] in ids]})
    out = {**src, "queries": rows, "max_candidates": k,
           "derived_from": "candidates_dev.json"}
    p = R / f"candidates_dev_cap{k}.json"
    p.write_text(json.dumps(out))
    npairs = sum(len(r["pairs"]) for r in rows)
    print(k, "queries", len(rows), "pairs", npairs, "->", p)
