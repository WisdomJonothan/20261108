"""Posthoc dev mechanism diagnostics: truncations of existing top100, not new retrieval."""
import json
from pathlib import Path
import numpy as np
from lib_reteco import read_trec_run
from ir_metrics import ndcg_at_k
R=Path(__file__).resolve().parents[1];S=R;out={'protocol':'Posthoc dev mechanism diagnostic. Fixed RRF k=60 and depths20/50/100. No dev-selected setting, no top300 retrieval; each depth truncates original dense top100. Complete qrels for IDCG.','arms':{}}
z=lambda a:(a-a.mean())/max(a.std(),1e-8)
for arm in ['title_ce','title_fusion','full_ce','full_fusion']:
 tag='fusion100_'+arm+'_s17_20261115';ref=json.loads((S/'results'/f'{tag}_dev_eval.json').read_text());records={str(n):{'domains':{}} for n in [20,50,100]}
 for dom in ref['domains']:
  def locate(name):
   for p in [R/'runs'/name,S/'runs'/name]:
    if p.is_file():return p
   raise FileNotFoundError(name)
  ce=read_trec_run(locate(f'{tag}_{dom}_dev_ce.trec'));dense=read_trec_run(locate(f'dense_mxbai_{dom}_dev.trec'));gold={}
  for line in (R/'data/track1_tempo'/dom/'qrels_dev.txt').read_text().splitlines():
   q,_,d,r=line.split();gold.setdefault(q,{})[d]=int(r)
  for n in [20,50,100]:
   qs={}
   for q in ref['domains'][dom]['queries']:
    rows=dense[q][:n];docs=[r[2] for r in rows];lookup={r[2]:r[1] for r in ce[q]};assert len(docs)==n and set(docs)<=set(lookup)
    ds=np.array([r[1] for r in rows]);cs=np.array([lookup[d] for d in docs]);co=np.argsort(-cs,kind='stable');cr=np.empty(n,int);cr[co]=np.arange(1,n+1)
    ranks={'ce':co,'z_half':np.argsort(-(z(ds)+z(cs)),kind='stable'),'rrf60':np.argsort(-(1/(60+np.arange(1,n+1))+1/(60+cr)),kind='stable')}
    qs[q]={k:float(ndcg_at_k([docs[i] for i in order],gold.get(q,{}),10)) for k,order in ranks.items()}
   records[str(n)]['domains'][dom]={'queries':qs,**{k:float(np.mean([v[k] for v in qs.values()])) for k in ['ce','z_half','rrf60']}}
 for n,v in records.items():
  assert len(v['domains'])==13 and sum(len(d['queries']) for d in v['domains'].values())==519
  v['macro']={k:float(np.mean([d[k] for d in v['domains'].values()])) for k in ['ce','z_half','rrf60']}
 assert abs(records['100']['macro']['z_half']-ref['macro']['fused'])<1e-12
 out['arms'][arm]=records
out['full_vs_title']={}
for training in ['ce','fusion']:
 out['full_vs_title'][training]={}
 for n in ['20','50','100']:
  f=out['arms']['full_'+training][n]['macro'];t=out['arms']['title_'+training][n]['macro'];delta={k:f[k]-t[k] for k in f}
  out['full_vs_title'][training][n]={'delta_full_minus_title':delta,'ce_better_but_z_worse':delta['ce']>0 and delta['z_half']<0,'ce_better_but_rrf_worse':delta['ce']>0 and delta['rrf60']<0}
out['title_calibration_external_report']={'alpha_ce':.7,'dev_macro':.3426963036740325,'provenance':'parent reports server result; no run manufactured or inferred','used_in_diagnostic':False}
p=R/'results/fusion100_posthoc_depth_rrf_diagnostic.json';p.write_text(json.dumps(out,indent=2));print(json.dumps(out['full_vs_title'],indent=2))
