"""Test position-balanced setwise decisions at the head of a frozen ranking."""
import argparse
from collections import defaultdict
from itertools import permutations
import json
from math import log
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_temporal_condition_intervention import evaluate
from replay_anchor_fallback import index
from score_specialist_shortlists import evidence


LABELS = 'ABCD'
ORDERS = tuple(permutations(range(3)))
ARMS = ('current', 'single_order', 'balanced_orders', 'confirmed_orders')
PROMPT = (
    'Which document gives the strongest evidence for one specific request in the question? '
    'Check the requested time, order of events and versions for the part it addresses. '
    'A document can answer one part without answering the whole question. '
    'Choose D if none of the documents offers usable evidence. Answer only A, B, C or D.'
)


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load(args.manifest)['queries'][args.shard_index::args.shards]
    reference = index(load(args.reference))
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain']].append(row)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', local_files_only=True)
    label_ids = [tokenizer.encode(c, add_special_tokens=False) for c in LABELS]
    if any(len(ids)!=1 for ids in label_ids) or len({ids[0] for ids in label_ids})!=len(LABELS):
        raise ValueError('The model requires distinct single-token labels')
    label_ids = [ids[0] for ids in label_ids]

    result = []
    started = perf_counter()
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            head = {r['qid']:reference[domain,r['qid']]['rankings']['verified_0.5'][:3] for r in group}
            bodies = read_documents(args.data_root/domain/'documents.jsonl',
                                    {doc for docs in head.values() for doc in docs})
            inputs = []
            for row in group:
                question = row['query']
                vocabulary = terms(question)
                snippets = [evidence(bodies[d],vocabulary,'dispersed') for d in head[row['qid']]]
                for order in ORDERS:
                    options = '\n\n'.join(f'Document {LABELS[i]}\n{snippets[j]}' for i,j in enumerate(order))
                    content = f'{PROMPT}\n\nQuestion\n{question}\n\n{options}'
                    chat = tokenizer.apply_chat_template([{'role':'user','content':content}], tokenize=False,
                                                          add_generation_prompt=True, enable_thinking=False)
                    inputs.append(chat+'Answer: ')
            logits = []
            for i in range(0,len(inputs),args.batch_size):
                tokens = tokenizer(inputs[i:i+args.batch_size], padding=True, add_special_tokens=False,
                                   return_tensors='pt').to('cuda')
                if tokens.input_ids.shape[1]>4096:
                    raise ValueError(f'Combined excerpts exceed the inference budget in {domain}')
                values = model(**tokens, use_cache=False, logits_to_keep=1).logits[:,-1,label_ids]
                logits.extend(values.float().cpu().tolist())
            for row, first in zip(group,range(0,len(logits),len(ORDERS)),strict=True):
                result.append({k:row[k] for k in ('qid','domain','split','cohort')} |
                              {'head':head[row['qid']],'logits':logits[first:first+len(ORDERS)]})
            print(domain,len(result),len(rows),round(perf_counter()-started),flush=True)
    return {'prompt':PROMPT,'model':args.model.name,'per_query':result}


def proposals(logits):
    aligned = np.array([np.array(row)[[order.index(j) for j in range(3)]+[3]]
                        for order,row in zip(ORDERS,logits,strict=True)])
    wins = aligned.argmax(axis=1)
    totals = aligned.mean(axis=0)
    best = int(totals.argmax())
    # Require a majority in the same document identity after changing its position.
    confident = best if (wins==best).sum()>len(ORDERS)/2 and totals[best]-totals[0]>=log(2) else 0
    return int(wins[0]), best, confident


def replay(args):
    source = index({'per_query':load(args.manifest)['queries']})
    reference = index(load(args.reference))
    shards = [load(path) for path in args.scores]
    if any(s['prompt']!=PROMPT for s in shards) or len({s['model'] for s in shards})!=1:
        raise ValueError('The shards must share the setwise prompt and checkpoint')
    by_query = index({'per_query':[r for shard in shards for r in shard['per_query']]})
    if not source.keys()==reference.keys()==by_query.keys():
        raise ValueError('Missing prompt scores for the frozen population')
    result = []
    for key,row in source.items():
        baseline = reference[key]['rankings']['verified_0.5']
        cached = by_query[key]
        if baseline[:3]!=cached['head']:
            raise ValueError(f'Candidate lists differ for {key}')
        rankings = {'current':baseline}
        for arm,i in zip(ARMS[1:],proposals(cached['logits']),strict=True):
            ranking = baseline.copy()
            if i<3:
                ranking[0],ranking[i] = ranking[i],ranking[0]
            rankings[arm] = ranking
        result.append({k:row[k] for k in ('domain','qid','cohort','split','actual_reference')} |
                      {'rankings':rankings})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    scoring, analysis = (modes.add_parser(name) for name in ('score','evaluate'))
    for mode in (scoring,analysis):
        for key in ('manifest','reference','data-root','output'):
            mode.add_argument(f'--{key}',type=Path,required=True)
    scoring.add_argument('--model',type=Path,required=True)
    for key,default in (('batch-size',10),('shards',1),('shard-index',0)):
        scoring.add_argument(f'--{key}',type=int,default=default)
    analysis.add_argument('--scores',type=Path,nargs='+',required=True)
    analysis.add_argument('--selection',type=Path)
    analysis.add_argument('--confirmation',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.mode=='score' and not (0<=args.shard_index<args.shards and args.batch_size>0):
        parser.error('Check shard and batch bounds')
    if args.mode=='evaluate' and args.confirmation and not args.selection:
        parser.error('A confirmation run requires a training-side selection')
    result = score(args) if args.mode=='score' else evaluate(args,replay,ARMS)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(result,file,separators=(',',':'))


if __name__=='__main__':
    main()
