import json, statistics as st
from collections import OrderedDict

p = r"D:\RETECO\github_repo\results\replacement_v2\query_reason_scores.json"
d = json.load(open(p))
views = ["orig", "reason", "analysis", "query_only"]
out = OrderedDict()
for v in views:
    vals = [d[k][v] for k in d if v in d[k]]
    out[v] = OrderedDict(mean=st.mean(vals), n=len(vals),
                         per_domain={k: d[k][v] for k in d})
print("domain-macro over %d domains" % len(d))
for v in views:
    print("%-11s %.4f" % (v, out[v]["mean"]))
best = max(views, key=lambda v: out[v]["mean"])
print("best", best, "vs orig %+.4f" % (out[best]["mean"] - out["orig"]["mean"]))
gain = {k: d[k]["reason"] - d[k]["orig"] for k in d}
print("reason - orig by domain:")
for k, v in gain.items():
    print("  %-10s %+.4f" % (k, v))
json.dump(out, open(r"D:\RETECO\github_repo\results\replacement_v2\query_reason_summary.json", "w"), indent=1)
print("wrote summary")
