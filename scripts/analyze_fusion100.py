"""Paired four-arm analysis. Run only after all 13-domain dev files are complete."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--draws',type=int,default=10000);args=ap.parse_args()
 arms=['title_ce','title_fusion','full_ce','full_fusion'];data={}
 for arm in arms:
  p=args.root/'results'/f'fusion100_{arm}_s17_20261115_dev_eval.json'
  data[arm]=json.loads(p.read_text(encoding='utf-8'))
  assert len(data[arm]['domains'])==13 and 'macro' in data[arm],f'Incomplete {arm}'
 domains=sorted(data[arms[0]]['domains']);mat={a:{} for a in arms}
 for d in domains:
  ids=sorted(data[arms[0]]['domains'][d]['queries'])
  for a in arms:
   assert ids==sorted(data[a]['domains'][d]['queries'])
   mat[a][d]=np.array([data[a]['domains'][d]['queries'][q]['fused'] for q in ids])
 assert sum(len(x) for x in mat[arms[0]].values())==519
 coeffs={'title_effect':[-1,1,0,0],'full_effect':[0,0,-1,1],'interaction':[1,-1,-1,1],'representation_independent':[-1,0,1,0]}
 report={'metric':'domain_macro_ndcg10','draws':args.draws,'seed':20261116,'systems':{},'contrasts':{}}
 for a in arms:
  recomputed=float(np.mean([v.mean() for v in mat[a].values()]));assert abs(recomputed-data[a]['macro']['fused'])<1e-10
  report['systems'][a]={'fused':recomputed,'ce':data[a]['macro']['ce']}
 for name,c in coeffs.items():
  delta={d:sum(c[i]*mat[a][d] for i,a in enumerate(arms)) for d in domains};means=np.array([delta[d].mean() for d in domains]);rng=np.random.default_rng(20261116)
  domain_boot=means[rng.integers(0,13,size=(args.draws,13))].mean(1)
  query_boot=np.zeros(args.draws)
  for d in domains:
   v=delta[d];query_boot+=v[rng.integers(0,len(v),size=(args.draws,len(v)))].mean(1)/13
  report['contrasts'][name]={'delta':float(means.mean()),'domain_ci95':np.quantile(domain_boot,[.025,.975]).tolist(),'within_domain_query_ci95':np.quantile(query_boot,[.025,.975]).tolist(),'per_domain':dict(zip(domains,means.tolist())),'positive_domains':int((means>0).sum()),'negative_domains':int((means<0).sum()),'without_iota':float(np.mean([v.mean() for d,v in delta.items() if d!='iota']))}
 p=args.root/'results/fusion100_four_arm_analysis.json';p.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
