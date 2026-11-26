"""Compare a pinned Rank1 reranker with and without its reasoning prefix."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_temporal_condition_intervention import evaluate, replay_views
from replay_anchor_cap import z
from replay_anchor_fallback import index
from score_specialist_shortlists import evidence


REVISION = '0b513ca87e0de16ffe17196c6746f28cafaad26b'
ARMS = ('current', 'rank1_direct_vote', 'rank1_reasoned_vote')


def truncate_trace(text):
    if '</think>' in text:
        return text.split('</think>', 1)[0].rstrip(), True
    text = text.rstrip()
    last = max(text.rfind(stop) for stop in ('.','!','?'))
    return text[:last+1] if last>=0 else text, False


def measure(prompts, model, tokenizer, answer_ids, budget):
    def logits(inputs):
        batch = tokenizer(inputs, padding=True, add_special_tokens=False,
                          return_tensors='pt').to(model.device)
        values = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, answer_ids].float()
        return (values[:,0]-values[:,1]).cpu().tolist()

    direct = logits([p+'\n</think>' for p in prompts])
    inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
    tokens = model.generate(**inputs, max_new_tokens=budget, do_sample=False, use_cache=True,
                            pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer,
                            stop_strings=['</think>'])
    suffix = tokens[:,inputs.input_ids.shape[1]:]
    traces = [truncate_trace(text.split(tokenizer.eos_token, 1)[0])
              for text in tokenizer.batch_decode(suffix, skip_special_tokens=False)]
    reasoned = logits([p+trace+'\n</think>' for p,(trace,_) in zip(prompts,traces,strict=True)])
    return list(zip(direct,reasoned,strict=True)), [complete for _,complete in traces]


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left',local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokens = [tokenizer.encode(t,add_special_tokens=False) for t in (' true',' false')]
    if any(len(t)!=1 for t in tokens) or tokens[0]==tokens[1]:
        raise ValueError('Expected distinct true and false tokens')
    ids = [t[0] for t in tokens]
    groups = defaultdict(list)
    for row in load(args.manifest)['queries'][args.shard_index::args.shards]:
        groups[row['domain']].append(row)
    started = perf_counter()
    result = []
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            docs = read_documents(args.data_root/domain/'documents.jsonl',
                                 {doc for row in group for doc in row['docs']})
            prompts = []
            for row in group:
                q = row['query']
                vocabulary = terms(q)
                for doc in row['docs']:
                    passage = evidence(docs[doc], vocabulary, 'dispersed')
                    prompts.append("Determine if the following passage is relevant to the query. "
                                   "Answer only with 'true' or 'false'.\n"
                                   f'Query: {q}\nPassage: {passage}\n<think>')
            scores, completes = [], []
            for i in range(0,len(prompts),args.batch_size):
                values, complete = measure(prompts[i:i+args.batch_size], model, tokenizer, ids, args.max_new_tokens)
                scores.extend(values)
                completes.extend(complete)
            cursor = 0
            for row in group:
                end = cursor + len(row['docs'])
                result.append({k:row[k] for k in ('qid','domain','split','cohort')} | {
                    'scores':dict(zip(row['docs'],scores[cursor:end],strict=True)),
                    'ended_thought':sum(completes[cursor:end])})
                cursor = end
            if cursor != len(scores):
                raise ValueError(f'Unassigned Rank1 scores in {domain}')
            print(domain,len(result),round(perf_counter()-started),flush=True)
    return {'model':args.model.name,'revision':REVISION,'max_new_tokens':args.max_new_tokens,
            'per_query':result}


def rank1_views(values):
    direct, reasoned = np.array(values).T
    return z(direct), z(reasoned)


def replay(args):
    shards = [load(p) for p in args.scores]
    if {s['revision'] for s in shards}!={REVISION} or len({s['max_new_tokens'] for s in shards})!=1:
        raise ValueError('Mismatched model weights or reasoning budgets')
    scores = index({'per_query':[row for s in shards for row in s['per_query']]})
    return replay_views(args,scores,ARMS,rank1_views)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    scoring, analysis = (modes.add_parser(name) for name in ('score','evaluate'))
    for command in (scoring,analysis):
        for key in ('manifest','data-root','output'):
            command.add_argument(f'--{key}',type=Path,required=True)
    scoring.add_argument('--model',type=Path,required=True)
    for key,value in (('batch-size',24),('shards',1),('shard-index',0),('max-new-tokens',192)):
        scoring.add_argument(f'--{key}',type=int,default=value)
    for key in ('consensus','reference','selection'):
        analysis.add_argument(f'--{key}',type=Path,required=key!='selection')
    for key in ('evidence','scores'):
        analysis.add_argument(f'--{key}',type=Path,required=True,nargs='+')
    analysis.add_argument('--confirmation',action='store_true')
    analysis.add_argument('--reserve-ids', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Provide a fresh output path')
    if args.mode=='score' and not (0<=args.shard_index<args.shards and min(args.batch_size,args.max_new_tokens)>0):
        parser.error('Check batch, shard and token bounds')
    if args.mode=='evaluate' and args.confirmation and not args.selection:
        parser.error('Confirmation requires a frozen rule')
    result=score(args) if args.mode=='score' else evaluate(args,replay,ARMS)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(result,file,separators=(',',':'))


if __name__=='__main__':
    main()
