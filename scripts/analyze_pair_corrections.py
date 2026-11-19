"""Fixed dense100 positive-negative pair diagnostics, conditional on qrels."""
import json,math,argparse
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
ap.add_argument('--experiment-root',type=Path)
ap.add_argument('--seed',type=int,default=17)
ap.add_argument('--run-template',default='fusion100_{arm}_s{seed}_{date}_{domain}_dev_{mode}.trec')
ap.add_argument('--date',default='20261115')
ap.add_argument('--arms',nargs='+',default=['title_ce','title_fusion','full_ce','full_fusion'])
ap.add_argument('--calibrated-json',type=Path)
ap.add_argument('--calibrated-run-template',help='Optional full CE calibrated TREC filename template, relative to experiment-root/runs')
ap.add_argument('--compare-report',type=Path,help='Previous seed pair report; compute query-paired difference of full-minus-title effects')
ap.add_argument('--output',type=Path)
args=ap.parse_args();R=args.root;E=args.experiment_root or R;arms=args.arms
assert 'title_ce' in arms and 'full_ce' in arms
calpath=args.calibrated_json or R/'results/full_ce_calibrated_dev_audited.json'
output=args.output or R/'results'/('pair_correction_analysis.json' if args.seed==17 else f'pair_correction_analysis_s{args.seed}.json')
def runpath(template,domain,arm='full_ce',mode='calibrated'):
 return E/'runs'/template.format(arm=arm,seed=args.seed,date=args.date,domain=domain,mode=mode)
def read(p):
 out={}
 for line in p.read_text().splitlines():
  q,_,d,r,s,t=line.split();out.setdefault(q,[]).append((int(r),d,float(s)))
 return {q:sorted(v) for q,v in out.items()}
def z(v):
 v=np.array(v);return (v-v.mean())/max(v.std(),1e-8)
def bs(ds):
 # Paired query resampling within domains; undefined ratios excluded.
 valid={d:np.array(v,float) for d,v in ds.items() if len(v)}
 if not valid:return None
 rng=np.random.default_rng(20261118);b=np.zeros(10000)
 for v in valid.values():b+=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)/len(valid)
 return {'macro':float(np.mean([v.mean() for v in valid.values()])), 'within_domain_query_ci95':np.quantile(b,[.025,.975]).tolist(),'eligible_domains':len(valid),'eligible_queries':sum(map(len,valid.values())),'per_domain':{d:float(v.mean()) for d,v in valid.items()}}
systems=[a+'_'+m for a in arms for m in ['ce','fused']]+['full_ce_calibrated'];rows=[];cal=json.loads(calpath.read_text());err=0
alpha=float(cal['alpha_ce']);assert 0<=alpha<=1
for domain in sorted(cal['domains']):
 dp=R/'data/track1_tempo'/domain;gold={};explicit={}
 for line in (dp/'qrels_dev.txt').read_text().splitlines():
  q,_,d,rel=line.split();explicit.setdefault(q,set()).add(d)
  if int(rel)>0:gold.setdefault(q,set()).add(d)
 dense=read(R/'runs'/f'dense_mxbai_{domain}_dev.trec');rr={a+'_'+m:read(runpath(args.run_template,domain,a,m)) for a in arms for m in ['ce','fused']}
 calibrated_run=read(runpath(args.calibrated_run_template,domain)) if args.calibrated_run_template else None
 ids=sorted(cal['domains'][domain]['queries']);assert set(ids)==set(gold)==set(dense)
 for q in ids:
  pool=[d for _,d,s in dense[q][:100]];dr={d:i for i,d in enumerate(pool)};g=gold[q];pos=[d for d in pool if d in g];neg=[d for d in pool if d not in g]
  ce=dict((d,s) for _,d,s in rr['full_ce_ce'][q]);cz=z([ce[d] for d in pool]);dz=z([s for _,d,s in dense[q][:100]])
  assert all(set(d for _,d,_ in rr[s][q])==set(pool) and len(rr[s][q])==100 for s in rr)
  order=sorted(range(100),key=lambda i:(-(alpha*cz[i]+(1-alpha)*dz[i]),pool[i]));calids=[pool[i] for i in order]
  if calibrated_run is not None:
   calids=[d for _,d,_ in calibrated_run[q]];assert len(calids)==100 and set(calids)==set(pool)
  den=sum(1/math.log2(i+2) for i in range(min(10,len(g))));v=sum((d in g)/math.log2(i+2) for i,d in enumerate(calids[:10]))/den
  err=max(err,abs(v-(cal['domains'][domain]['queries'][q]['calibrated'] if isinstance(cal['domains'][domain]['queries'][q],dict) else cal['domains'][domain]['queries'][q])));ranks={s:{d:i for i,(_,d,_) in enumerate(rr[s][q])} for s in systems[:-1]};ranks['full_ce_calibrated']={d:i for i,d in enumerate(calids)}
  for subset in ['all','dense_top10_touch','dense_top10_crossing']:
   pairs=[(p,n) for p in pos for n in neg if subset=='all' or (min(dr[p],dr[n])<10 if subset=='dense_top10_touch' else (dr[p]<10)!=(dr[n]<10))]
   wrong=[(p,n) for p,n in pairs if dr[p]>dr[n]];right=[(p,n) for p,n in pairs if dr[p]<dr[n]]
   item={'domain':domain,'query':q,'subset':subset,'pair_count':len(pairs),'dense_wrong_count':len(wrong),'dense_right_count':len(right),'unannotated_pool_as_zero':sum(d not in explicit[q] for d in neg),'systems':{}}
   item['dense_error_rate']=len(wrong)/len(pairs) if pairs else None
   for s,sr in ranks.items():
    fix=sum(sr[p]<sr[n] for p,n in wrong);breaks=sum(sr[p]>sr[n] for p,n in right)
    item['systems'][s]={'correction_rate':fix/len(wrong) if wrong else None,'destruction_rate':breaks/len(right) if right else None,'net_pair_gain':(fix-breaks)/len(pairs) if pairs else None,'corrected':fix,'destroyed':breaks}
   rows.append(item)
assert len(rows)==519*3 and err<1e-10
out={'seed':args.seed,'alpha_ce':alpha,'calibrated_json':str(calpath),'run_template':args.run_template,'queries':519,'domains':13,'calibrated_reconstruction_max_ndcg_error':err,'scope':'Benchmark relevance: qrels-positive versus all other dense100 documents. Unannotated documents are treated as zero, not asserted truly irrelevant. All pairs use fixed dense ranking; top10_touch includes either document in dense top10, top10_crossing includes exactly one. Rank ties follow TREC ordering. Undefined zero-denominator query ratios excluded and eligible counts reported. Query-bootstrap conditional on trained single-seed models.','summary':{},'paired_full_minus_title':{},'per_query':rows}
for subset in ['all','dense_top10_touch','dense_top10_crossing']:
 selected=[r for r in rows if r['subset']==subset];domains=sorted({r['domain'] for r in selected});out['summary'][subset]={}
 out['summary'][subset]['dense_error_rate']=bs({d:[r['dense_error_rate'] for r in selected if r['domain']==d and r['dense_error_rate'] is not None] for d in domains})
 for s in systems:
  out['summary'][subset][s]={m:bs({d:[r['systems'][s][m] for r in selected if r['domain']==d and r['systems'][s][m] is not None] for d in domains}) for m in ['correction_rate','destruction_rate','net_pair_gain']}
 out['paired_full_minus_title'][subset]={}
 for mode in ['ce','fused']:
  a='full_ce_'+mode;b='title_ce_'+mode
  out['paired_full_minus_title'][subset][mode]={m:bs({d:[r['systems'][a][m]-r['systems'][b][m] for r in selected if r['domain']==d and r['systems'][a][m] is not None] for d in domains}) for m in ['correction_rate','destruction_rate','net_pair_gain']}
if args.compare_report:
 prior=json.loads(args.compare_report.read_text());old={(r['domain'],r['query'],r['subset']):r for r in prior['per_query']}
 assert set(old)=={(r['domain'],r['query'],r['subset']) for r in rows}
 out['cross_seed']={'reference_seed':prior.get('seed',17),'new_seed':args.seed,'scope':'Difference of full-minus-title effects, query-paired conditional on two checkpoints; two seeds are not an estimate of population training variance.','difference_of_effects':{},'reference_full_minus_title':prior['paired_full_minus_title']}
 for subset in ['all','dense_top10_touch','dense_top10_crossing']:
  selected=[r for r in rows if r['subset']==subset];out['cross_seed']['difference_of_effects'][subset]={}
  for mode in ['ce','fused']:
   a='full_ce_'+mode;b='title_ce_'+mode;stats={}
   for m in ['correction_rate','destruction_rate','net_pair_gain']:
    ds={d:[] for d in domains}
    for r in selected:
     prev=old[r['domain'],r['query'],subset]
     assert (r['pair_count'],r['dense_wrong_count'],r['dense_right_count'])==(prev['pair_count'],prev['dense_wrong_count'],prev['dense_right_count'])
     if r['systems'][a][m] is not None:
      ds[r['domain']].append((r['systems'][a][m]-r['systems'][b][m])-(prev['systems'][a][m]-prev['systems'][b][m]))
    stats[m]=bs(ds)
   out['cross_seed']['difference_of_effects'][subset][mode]=stats
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out['paired_full_minus_title'],indent=2))
