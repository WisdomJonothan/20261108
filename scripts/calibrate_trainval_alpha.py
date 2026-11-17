"""Fixed-grid train-heldout calibration; never reads official dev."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ir_metrics import ndcg_at_k
ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);ap.add_argument('--split',default='results/fusion100_split.json');a=ap.parse_args()
p=json.loads(Path(a.input).read_text());assert p['split']=='train_heldout'
sp=json.loads(Path(a.split).read_text())['eval'];expected={(d,q) for d,qs in sp.items() for q in qs};rows=p['rows']
assert len(rows)==214 and {(r['domain'],r['qid']) for r in rows}==expected
z=lambda x:(x-x.mean())/max(x.std(),1e-8)
grid=[]
for tenth in range(11):
 alpha=tenth/10;domains={d:[] for d in sp}
 for r in rows:
  ce=np.asarray(r['ce']);dense=np.asarray(r['dense']);assert len(ce)==len(dense)==len(r['docs'])==100 and np.isfinite(ce).all() and np.isfinite(dense).all()
  scores=alpha*z(ce)+(1-alpha)*z(dense);order=np.argsort(-scores,kind='stable');rank=[r['docs'][i] for i in order]
  domains[r['domain']].append(float(ndcg_at_k(rank,r['gold'],10)))
 per={d:float(np.mean(v)) for d,v in domains.items()};grid.append({'alpha_ce':alpha,'macro':float(np.mean(list(per.values()))),'domains':per})
# Exact ties prefer the original 1:1 weight, then the smaller alpha.
best=min(grid,key=lambda r:(-r['macro'],abs(r['alpha_ce']-.5),r['alpha_ce']))
Path(a.output).write_text(json.dumps({'tag':p['tag'],'protocol':'internal heldout selection, not an unbiased gain estimate','grid':grid,'selected':best,'delta_vs_half':best['macro']-grid[5]['macro']},indent=2));print(json.dumps(best))
