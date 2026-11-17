"""Evaluate frozen train-validation alpha; 0.5 is a reproduction check only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from lib_reteco import read_trec_run,write_trec_run
from ir_metrics import ndcg_at_k
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--alpha-json',type=Path,required=True);ap.add_argument('--original-eval',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
frozen=json.loads(a.alpha_json.read_text());tag=frozen['tag'];alpha=float(frozen['selected']['alpha_ce']);assert 0<=alpha<=1
assert frozen['protocol']=='internal heldout selection, not an unbiased gain estimate'
expected_best=min(frozen['grid'],key=lambda r:(-r['macro'],abs(r['alpha_ce']-.5),r['alpha_ce']))
assert expected_best==frozen['selected']
old=json.loads(a.original_eval.read_text());assert old['tag']==tag
assert len(old['domains'])==13
z=lambda x:(x-x.mean())/max(x.std(),1e-8)
out={'tag':tag,'alpha_ce':alpha,'alpha_source_sha256':hashlib.sha256(a.alpha_json.read_bytes()).hexdigest(),'domains':{}};pending=[];count=0
for dom,ref in old['domains'].items():
 data=a.root/'data/track1_tempo'/dom
 queries=[json.loads(x)['id'] for x in (data/'examples_dev.jsonl').read_text().splitlines() if x.strip()]
 gold={}
 for line in (data/'qrels_dev.txt').read_text().splitlines():
  q,_,d,r=line.split();gold.setdefault(q,{})[d]=int(r)
 dense=read_trec_run(a.root/'runs'/f'dense_mxbai_{dom}_dev.trec');ce=read_trec_run(a.root/'runs'/f'{tag}_{dom}_dev_ce.trec');assert set(queries)==set(ref['queries'])
 vals=[];checks=[];perq={};run={}
 for q in queries:
  dr=dense[q][:100];docs=[x[2] for x in dr];csmap={x[2]:x[1] for x in ce[q]};assert len(docs)==len(set(docs))==100 and set(docs)==set(csmap)
  ds=np.array([x[1] for x in dr]);cs=np.array([csmap[d] for d in docs]);assert np.isfinite(cs).all() and np.isfinite(ds).all()
  zd,zc=z(ds),z(cs)
  half=np.argsort(-(zd+zc),kind='stable');check=float(ndcg_at_k([docs[i] for i in half],gold.get(q,{}),10))
  assert abs(check-ref['queries'][q]['fused'])<1e-12,('0.5 reproduction failed; inspect six-decimal TREC precision',dom,q,check,ref['queries'][q]['fused'])
  score=alpha*zc+(1-alpha)*zd;order=np.argsort(-score,kind='stable');rank=[docs[i] for i in order];value=float(ndcg_at_k(rank,gold.get(q,{}),10))
  vals.append(value);checks.append(check);perq[q]={'calibrated':value,'half':check};run[q]=[(docs[i],float(score[i])) for i in order]
 out['domains'][dom]={'n':len(queries),'calibrated':float(np.mean(vals)),'half':float(np.mean(checks)),'queries':perq};count+=len(queries);pending.append((dom,run))
assert count==519
out['coverage']={'queries':count,'domains':len(out['domains'])};out['macro']={k:float(np.mean([d[k] for d in out['domains'].values()])) for k in ['calibrated','half']}
assert abs(out['macro']['half']-old['macro']['fused'])<1e-12
out['delta']=out['macro']['calibrated']-out['macro']['half'];out['half_reproduction']='passed for every query'
for dom,run in pending:write_trec_run(str(a.root/'runs'/f'{tag}_{dom}_dev_calibrated.trec'),run,tag=tag+'_calibrated')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2));print(json.dumps({'alpha':alpha,'coverage':out['coverage'],'macro':out['macro'],'delta':out['delta']}))
