import json, math, hashlib
from pathlib import Path
import numpy as np
R=Path(r'D:\RETECO')
new=json.loads((R/'results/full_ce_calibrated_dev.json').read_text())
old=json.loads((R/'results/server_review_20261116/results/fusion100_full_ce_s17_20261115_dev_eval.json').read_text())
out={'alpha_ce':new['alpha_ce'],'selection':new['selection'],'scope':'Single trained seed; alpha fixed on internal validation before official dev. Bootstrap conditions on checkpoint and selected alpha; excludes training and calibration-selection uncertainty. Frozen comparison changes training recipe.','comparisons':{},'gold_sources':{}}
values={k:{} for k in ['new','full_ce_equal','frozen']}
def score(ids,g):
 den=sum(1/math.log2(i+2) for i in range(min(10,len(g))))
 return sum((d in g)/math.log2(i+2) for i,d in enumerate(ids[:10]))/den if den else 0
for d,entry in new['domains'].items():
 p=R/'data/track1_tempo'/d/'qrels_dev.txt';gold={}
 for line in p.read_text().splitlines():
  q,_,doc,rel=line.split()
  if int(rel)>0:gold.setdefault(q,set()).add(doc)
 ids=sorted(json.loads(l)['id'] for l in (p.parent/'examples_dev.jsonl').read_text(encoding='utf-8').splitlines())
 assert set(ids)==set(entry['queries'])==set(old['domains'][d]['queries'])==set(gold)
 rows={}
 for line in (R/'runs'/f'sub_sft_e2bc_{d}_dev.trec').read_text().splitlines():
  q,_,doc,r,s,t=line.split();rows.setdefault(q,[]).append((int(r),doc))
 values['new'][d]=np.array([entry['queries'][q] for q in ids])
 values['full_ce_equal'][d]=np.array([old['domains'][d]['queries'][q]['fused'] for q in ids])
 values['frozen'][d]=np.array([score([doc for _,doc in sorted(rows[q])],gold[q]) for q in ids])
 out['gold_sources'][d]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n':len(ids)}
 assert abs(values['new'][d].mean()-entry['ndcg10'])<1e-12
assert len(values['new'])==13 and sum(map(len,values['new'].values()))==519
out['queries']=519;out['domains']=13
out['macros']={k:float(np.mean([a.mean() for a in v.values()])) for k,v in values.items()}
for name in ['full_ce_equal','frozen']:
 delta={d:values['new'][d]-v for d,v in values[name].items()};means=np.array([v.mean() for v in delta.values()]);rng=np.random.default_rng(20261118)
 db=means[rng.integers(0,13,(10000,13))].mean(1);qb=np.zeros(10000)
 for v in delta.values():qb+=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)/13
 out['comparisons'][name]={'delta':float(means.mean()),'domain_ci95':np.quantile(db,[.025,.975]).tolist(),'within_domain_query_ci95':np.quantile(qb,[.025,.975]).tolist(),'per_domain':{d:float(v.mean()) for d,v in delta.items()},'positive_domains':int((means>0).sum()),'negative_domains':int((means<0).sum()),'recovered_zero':sum(int(((values[name][d]==0)&(values['new'][d]>0)).sum()) for d in delta),'new_zero':sum(int(((values[name][d]>0)&(values['new'][d]==0)).sum()) for d in delta)}
out['zero_counts']={k:sum(int((v==0).sum()) for v in ds.values()) for k,ds in values.items()}
out['bootstrap']={'draws':10000,'seed':20261118,'paired':True}
(R/'results/full_ce_calibrated_statistics_review.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='gold_sources'},indent=2))
