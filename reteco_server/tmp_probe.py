
import json
from collections import Counter
rows=[]
for f in ("f0","f1","f2"):
    d=json.load(open(f"results/replacement_v2/expanded3_fixed_train_{f}.json"))
    for r in d["budgets"]["18"]["per_query"]: rows.append(r)
tally=Counter(); veto=Counter()
for r in rows:
    pool=sorted(r["actions"], key=lambda a:(a["slot"], -a["prior"]))[:6]
    elig=[a for a in pool if a["agree"] and a["mp"]>0.5]
    best=max(elig,key=lambda a:a["mp"]) if elig else None
    bb=max(pool,key=lambda a:a["delta"])
    if best and best["delta"]>1e-9: tally["positive"]+=1
    elif best and best["delta"]<-1e-9: tally["harmful"]+=1
    elif bb["delta"]>1e-9:
        tally["missed"]+=1
        if bb["mp"]>0.5 and not bb["agree"]: veto["vetoed_agree"]+=1
        elif bb["mp"]>0.5: veto["mp_gt_half_but_no_agree"]+=1
        elif bb["mp"]>0.35: veto["mp_0.35_0.5"]+=1
        else: veto["confident_reject"]+=1
    else: tally["neutral"]+=1
print("tally",dict(tally)); print("missed breakdown",dict(veto))
