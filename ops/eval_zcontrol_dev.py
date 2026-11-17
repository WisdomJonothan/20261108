import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/"scripts"))
from train_ce_zcontrol import clean_query,split_raw_query,compose_query,window_texts
from lib_reteco import read_trec_run,write_trec_run
from ir_metrics import ndcg_at_k
R=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description="Explicit post-completion dev evaluation; never invoked by training.")
ap.add_argument('--tag',default='fusion100_full_zce_s17_20261116')
ap.add_argument('--confirm-dev-evaluation',action='store_true',required=True)
args=ap.parse_args()
if Path(args.tag).name!=args.tag:raise ValueError('tag must be a basename')
for tag in [args.tag]:
 base=R/'results/ce_ft'/tag
 # A saved best can exist while training is still running: require the final done event.
 logs=[base/'train_log.jsonl'] if (base/'train_log.jsonl').is_file() else []
 done=False
 for logpath in logs:
  for line in logpath.read_text().splitlines():
   try: event=json.loads(line)
   except ValueError: continue
   if event.get('event')=='done':done=True
 if not done:raise RuntimeError('Training done event missing; dev evaluation refused')
 if (R/'results'/f'{tag}_dev_eval.json').exists():raise FileExistsError('Dev evaluation already exists; inspect it before explicitly rerunning')
 cfg=json.loads((base/'config.json').read_text()); assert (base/'best').is_dir()
 tok=AutoTokenizer.from_pretrained(base/'best'); model=AutoModelForSequenceClassification.from_pretrained(base/'best',dtype=torch.bfloat16).eval().cuda()
 output={'tag':tag,'config':cfg,'domains':{}}
 for dom in cfg['domains']:
  data=R/'data/track1_tempo'/dom; queries=[json.loads(x) for x in (data/'examples_dev.jsonl').read_text().splitlines() if x.strip()]
  run=read_trec_run(R/'runs'/f'dense_mxbai_{dom}_dev.trec'); needed={x[2] for q in queries for x in run[q['id']][:100]}; corpus={}
  with (data/'documents.jsonl').open() as f:
   for line in f:
    d=json.loads(line)
    if d['id'] in needed:corpus[d['id']]=d['content']
  assert needed<=corpus.keys()
  gold={}
  for line in (data/'qrels_dev.txt').read_text().splitlines():
   q,_,d,rel=line.split();gold.setdefault(q,{})[d]=int(rel)
  ce_run={}; fused_run={}; metrics={}
  with torch.inference_mode():
   for i,q in enumerate(queries):
    qid=q['id'];title,body=split_raw_query(q['query']);qt=compose_query(clean_query(title),clean_query(body),cfg['q_mode'],cfg['q_cap_words']);qt=tok.decode(tok(qt,add_special_tokens=False)['input_ids'][:cfg['q_cap']])
    rows=run[qid][:100];docs=[x[2] for x in rows];flat=[];spans=[]
    for d in docs:
     ws=window_texts(tok,corpus[d],cfg['win'],cfg['stride']);spans.append((len(flat),len(flat)+len(ws)));flat.extend(ws)
    scores=[]
    for j in range(0,len(flat),cfg['pool_chunk']):
     chunk=flat[j:j+cfg['pool_chunk']];enc=tok([qt]*len(chunk),chunk,padding=True,truncation=True,max_length=cfg['max_len'],return_tensors='pt').to('cuda');scores.extend(model(**enc).logits.flatten().float().cpu().tolist())
    cs=np.array([max(scores[a:b]) for a,b in spans]);ds=np.array([x[1] for x in rows]);fs=(cs-cs.mean())/max(cs.std(),1e-8)+(ds-ds.mean())/max(ds.std(),1e-8)
    co=np.argsort(-cs,kind='stable');fo=np.argsort(-fs,kind='stable');ce_run[qid]=[(docs[k],float(cs[k])) for k in co];fused_run[qid]=[(docs[k],float(fs[k])) for k in fo]
    metrics[qid]={'ce':ndcg_at_k([docs[k] for k in co],gold.get(qid,{}),10),'fused':ndcg_at_k([docs[k] for k in fo],gold.get(qid,{}),10)}
    if (i+1)%10==0:print(tag,dom,i+1,len(queries),flush=True)
  write_trec_run(str(R/'runs'/f'{tag}_{dom}_dev_ce.trec'),ce_run,tag=tag)
  write_trec_run(str(R/'runs'/f'{tag}_{dom}_dev_fused.trec'),fused_run,tag=tag)
  output['domains'][dom]={'queries':metrics,'ce':float(np.mean([x['ce'] for x in metrics.values()])),'fused':float(np.mean([x['fused'] for x in metrics.values()]))}
  (R/'results'/f'{tag}_dev_eval.json').write_text(json.dumps(output,indent=2))
 assert len(output['domains'])==13 and sum(len(d['queries']) for d in output['domains'].values())==519
 output['macro']={k:float(np.mean([d[k] for d in output['domains'].values()])) for k in ['ce','fused']}
 (R/'results'/f'{tag}_dev_eval.json').write_text(json.dumps(output,indent=2));print(tag,output['macro'],flush=True)
 del model;torch.cuda.empty_cache()
