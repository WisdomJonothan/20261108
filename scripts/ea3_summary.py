"""Summarise the expanded-action cache and the position-aware gate into one artefact."""
import json, os
import numpy as np
from collections import defaultdict

ROOT = r"D:\RETECO\github_repo"
V2 = os.path.join(ROOT, "results", "replacement_v2")
d = json.load(open(os.path.join(V2, "expanded3_prior_dev.json")))

rows = []
seen = set()
for k, b in d["budgets"].items():
    for q in b["per_query"]:
        for a in q.get("actions", []):
            key = (q["qid"], a["slot"], a["challenger"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(slot=a["slot"], delta=a["delta"], mp=a["mp"], agree=a["agree"]))

by_slot = defaultdict(list)
for r in rows:
    by_slot[r["slot"]].append(r)
slots = {}
for s in sorted(by_slot):
    v = by_slot[s]
    slots[str(s)] = dict(
        judged=len(v),
        helpful=sum(1 for r in v if r["delta"] > 1e-9),
        harmful=sum(1 for r in v if r["delta"] < -1e-9),
        mean_delta=sum(r["delta"] for r in v) / len(v),
    )

budgets = {}
for k, b in sorted(d["budgets"].items(), key=lambda kv: int(kv[0])):
    budgets[k] = dict(macro_policy=b["macro_policy"], macro_delta=b["macro_delta"],
                      ci95=b["bootstrap"]["ci95"], p_le_0=b["bootstrap"]["p_le_0"],
                      swap_counts=b["swap_counts"])

out = dict(
    source="expanded3_prior_dev.json",
    n_queries=d["n_queries"], max_actions=d["max_actions"],
    prompts=d["prompts"], seconds=d["seconds"], gpu_peak_mb=d["gpu_peak_mb"],
    budgets=budgets,
    action_pool=dict(
        actions_judged=len(rows),
        candidates="union(dense top-100, frozen full-query CE top-100) minus frozen top-10, best-rank order, 20 kept",
        slots="1..10",
    ),
    by_slot=slots,
    note=("Widening the pool and the slot range lowers the score monotonically: the judge accepts "
          "more swaps but the extra ones are wrong. Replacing a top-3 document costs 10-40x more "
          "than replacing a tail document (mean delta -0.083 at slot 1) while the judge's "
          "reliability does not improve with position."),
)
with open(os.path.join(V2, "expanded3_analysis_dev.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out["budgets"][k]["macro_delta"] for k in out["budgets"]}, indent=0))
print("actions", len(rows))
