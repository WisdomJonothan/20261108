"""Test whether a stronger judge consistently identifies a better first result."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_temporal_condition_intervention import evaluate
from replay_anchor_fallback import index
from replay_passage_admission import observations
from score_specialist_shortlists import evidence


ARMS = ('current','balanced_direct','balanced_explained')
DEBIASED = ('current','difference_direct','difference_explained')
RUBRIC = (
    'Compare these two documents as evidence for the question. A document may answer one '
    'specific request. Check that the relevant event, time or version matches that request, '
    'and distinguish a stated answer from topical overlap. Briefly identify relevant '
    'evidence from each document and any explicit incompatibility. Choose A or B.'
)


def prepare(args):
    rows = index({'per_query':load(args.manifest)['queries']})
    base, votes = (index(load(p)) for p in (args.reference,args.consensus))
    known = observations(args.evidence)
    if not rows.keys()==base.keys()==votes.keys()==known.keys():
        raise ValueError('Pair selection requires exactly the frozen query set')
    output = []
    for key,row in rows.items():
        order = base[key]['rankings']['verified_0.5']
        pointwise = votes[key]['rankings']['pointwise']
        candidates = [d for d in pointwise if d in order[1:5] and d in known[key]
                      and known[key][d]['verified'] and known[key][d]['reasoned']>=2]
        pair = [order[0],candidates[0]] if candidates else []
        output.append(row | {'docs':pair, 'current':order})
    print('Eligible queries',sum(bool(r['docs']) for r in output),'of',len(output),flush=True)
    return {'queries':output}


def score(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True,padding_side='left')
    model = AutoModelForImageTextToText.from_pretrained(args.model,local_files_only=True,
                                                       dtype=torch.bfloat16).to('cuda').eval()
    labels = [tokenizer.encode(c,add_special_tokens=False) for c in ('A','B')]
    if any(len(t)!=1 for t in labels) or labels[0]==labels[1]:
        raise ValueError('Expected distinct answer tokens')
    letters = [t[0] for t in labels]
    rows = load(args.manifest)['queries'][args.shard_index::args.shards]
    groups = defaultdict(list)
    for r in rows:
        groups[r['domain']].append(r)
    output = []
    start = perf_counter()
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            collection = {d for r in group for d in r['docs']}
            texts = read_documents(args.data_root/domain/'documents.jsonl',collection) if collection else {}
            active = []
            for row in group:
                output.append({'qid':row['qid'],'domain':domain,'docs':row['docs'],'scores':[]})
                if not row['docs']:
                    continue
                question = row['query']
                excerpts = [evidence(texts[d],terms(question),'dispersed') for d in row['docs']]
                for first,second in (excerpts,list(reversed(excerpts))):
                    message = f'{RUBRIC}\n\nQuestion\n{question}\n\nA\n{first}\n\nB\n{second}'
                    prefix = tokenizer.apply_chat_template(
                        [{'role':'user','content':message}],tokenize=False,
                        add_generation_prompt=True,enable_thinking=False)
                    active.append((output[-1]['scores'],prefix))
            for offset in range(0,len(active),args.batch_size):
                block = active[offset:offset+args.batch_size]
                prefixes = [prefix for _,prefix in block]
                input_tokens = tokenizer(prefixes,padding=True,add_special_tokens=False,
                                         return_tensors='pt').to('cuda')
                if input_tokens.input_ids.shape[-1]+args.max_new_tokens+8>4096:
                    raise ValueError(f'Pair comparison exceeded the bounded input in {domain}')
                direct = tokenizer([p+'Choice: ' for p in prefixes],padding=True,add_special_tokens=False,
                                   return_tensors='pt').to('cuda')
                raw = model(**direct,use_cache=False,logits_to_keep=1).logits[:,-1,letters].float()
                generated = model.generate(**input_tokens,do_sample=False,top_k=None,top_p=None,temperature=None,
                                           max_new_tokens=args.max_new_tokens,pad_token_id=tokenizer.pad_token_id)
                replies = tokenizer.batch_decode(generated[:,input_tokens.input_ids.shape[-1]:],
                                                 skip_special_tokens=True)
                explained = tokenizer([p+r+'\nChoice: ' for p,r in zip(prefixes,replies,strict=True)],
                                      padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
                proof = model(**explained,use_cache=False,logits_to_keep=1).logits[:,-1,letters].float()
                margins = torch.stack((raw[:,1]-raw[:,0],proof[:,1]-proof[:,0]),dim=1).cpu().tolist()
                for (target,_),values in zip(block,margins,strict=True):
                    target.append(values)
            print(domain,len(output),len(rows),len(active),round(perf_counter()-start),flush=True)
    return {'model':args.model.name,'rubric':RUBRIC,'per_query':output}


def replay(args):
    source = index({'per_query':load(args.manifest)['queries']})
    shards = [load(p) for p in args.scores]
    if len({(s['model'],s['rubric']) for s in shards})!=1 or any(s['rubric']!=RUBRIC for s in shards):
        raise ValueError('Every pair score must use the same pinned instruction')
    measured = index({'per_query':[r for s in shards for r in s['per_query']]})
    if measured.keys()!=source.keys():
        raise ValueError('Pair scores must cover the complete matched population')
    output = []
    for key,row in source.items():
        observed = measured[key]
        pair, prior = row['docs'], row['current']
        if observed['docs']!=pair or len(observed['scores'])!=len(pair):
            raise ValueError(f'Missing ordered pair scores for {key}')
        rankings = {'current':prior}
        methods = DEBIASED if args.remove_position_bias else ARMS
        for i,method in enumerate(methods[1:]):
            edited = prior.copy()
            if pair:
                forward, reverse = (r[i] for r in observed['scores'])
                replace = forward > reverse if args.remove_position_bias else forward > 0 > reverse
                if replace:
                    other = prior.index(pair[1])
                    edited[0],edited[other]=edited[other],edited[0]
            rankings[method]=edited
        output.append({k:row[k] for k in ('domain','qid','actual_reference','split','cohort')} |
                      {'rankings':rankings})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    stages=p.add_subparsers(dest='stage',required=True)
    setup,scoring,comparison=(stages.add_parser(name) for name in ('prepare','score','evaluate'))
    for mode in (setup,scoring,comparison):
        for key in ('manifest','output'):
            mode.add_argument(f'--{key}',type=Path,required=True)
    for key in ('reference','consensus'):
        setup.add_argument(f'--{key}',type=Path,required=True)
    setup.add_argument('--evidence',type=Path,nargs='+',required=True)
    for key in ('model','data-root'):
        scoring.add_argument(f'--{key}',type=Path,required=True)
    for key,default in (('shards',1),('shard-index',0),('batch-size',2),('max-new-tokens',128)):
        scoring.add_argument(f'--{key}',type=int,default=default)
    for key in ('data-root','selection','reserve-ids'):
        comparison.add_argument(f'--{key}',type=Path,required=key=='data-root')
    comparison.add_argument('--scores',type=Path,nargs='+',required=True)
    comparison.add_argument('--confirmation',action='store_true')
    comparison.add_argument('--remove-position-bias',action='store_true',
                            help='Use the signed difference between the two presentation logits')
    args=p.parse_args()
    if args.output.exists() or (args.stage=='evaluate' and args.confirmation and not args.selection):
        p.error('Use a fresh output and lock the policy before confirmation')
    if args.stage=='score' and not (0<=args.shard_index<args.shards and min(args.batch_size,args.max_new_tokens)>0):
        p.error('Check the shard and inference budget')
    if args.stage=='evaluate':
        report=evaluate(args,replay,DEBIASED if args.remove_position_bias else ARMS)
    else:
        report={'prepare':prepare,'score':score}[args.stage](args)
    with args.output.open('x',encoding='utf8') as out:
        json.dump(report,out,separators=(',',':'))


if __name__=='__main__':
    main()
