import json, os, sys
from collections import Counter, defaultdict
ROOT='/homeC/wujiaheng/RETECO'
DOMAINS=[d for d in sorted(os.listdir(os.path.join(ROOT,'data/track1_tempo'))) if os.path.exists(os.path.join(ROOT,'data/track1_tempo',d,'qrels_train.txt')) and os.path.exists(os.path.join(ROOT,'runs',f'dense_mxbai_{d}_train.trec'))]
def load_qrels(d):
    q=defaultdict(dict)
    for ln in open(os.path.join(ROOT,'data/track1_tempo',d,'qrels_train.txt'),encoding='utf-8'):
        p=ln.split()
        if len(p)>=3: q[p[0]][p[-2] if p[0]==p[0] else p[2]]=int(p[-1])
    return q
out={}
for d in DOMAINS:
    qf=os.path.join(ROOT,'data/track1_tempo',d,'qrels_train.txt')
    rel=defaultdict(dict)
    for ln in open(qf,encoding='utf-8'):
        p=ln.split()
        if len(p)==3: rel[p[0]][p[1]]=int(p[2])
        elif len(p)==4: rel[p[0]][p[2]]=int(p[3])
    rp=os.path.join(ROOT,'runs',f'dense_mxbai_{d}_train.trec')
    run=defaultdict(list)
    for ln in open(rp,encoding='utf-8'):
        p=ln.split()
        if len(p)>=4: run[p[0]].append(p[2])
    judged0=unjudged=pos=0; hard_unj=hard_tot=0
    for qid,docs in run.items():
        r=rel.get(qid,{})
        cand=docs[:100]
        p=[x for x in cand if r.get(x,0)>0]
        pos+=len(p)
        negs=[x for x in cand if r.get(x,0)<=0]
        j0=[x for x in negs if x in r]; un=[x for x in negs if x not in r]
        judged0+=len(j0); unjudged+=len(un)
        n_neg=max(0,20-len(p)); n_hard=int(round(n_neg*0.7))
        hard_no_j0=[x for x in j0][:n_hard]
        # hard negatives are taken from the top of neg_pool (judged-0 and unjudged interleaved by rank)
        head=negs[:n_hard]
        hard_tot+=len(head); hard_unj+=len([x for x in head if x not in r])
    out[d]=dict(queries=len(run),positives=pos,neg_judged0=judged0,neg_unjudged=unjudged,
                hard_total=hard_tot,hard_unjudged=hard_unj)
tot=Counter()
for d,v in out.items():
    for k in ('queries','positives','neg_judged0','neg_unjudged','hard_total','hard_unjudged'): tot[k]+=v[k]
out['_total']=dict(tot)
out['_share']=dict(neg_unjudged=round(tot['neg_unjudged']/(tot['neg_unjudged']+tot['neg_judged0']),4),
                   hard_unjudged=round(tot['hard_unjudged']/max(1,tot['hard_total']),4))
print(json.dumps(out,indent=1)[:2000])
open('/homeC/wujiaheng/RETECO/results/analysis/neg_audit.json','w').write(json.dumps(out,indent=1))
