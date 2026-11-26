"""Evaluate a reasoning retriever as an additional fixed-pool evidence view."""
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
from eval_temporal_evidence_categories import question_facets
from replay_anchor_cap import z
from replay_anchor_fallback import index
from run_dense_generic import encode
from score_specialist_shortlists import evidence


REVISION = 'ff7f8bc8d1827734dcf2bbfab99f065be618069a'
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
ARMS = ('current', 'diver_full_vote', 'diver_clause_vote')


def score(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16,
                                     local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='right', local_files_only=True)
    groups = defaultdict(list)
    for row in load(args.manifest)['queries'][args.shard_index::args.shards]:
        groups[row['domain']].append(row)
    result = []
    started = perf_counter()
    for domain, group in sorted(groups.items()):
        docs = read_documents(args.data_root/domain/'documents.jsonl',
                              {doc for row in group for doc in row['docs']})
        items, spans, questions = {}, {}, {}
        for row in group:
            question = row['query']
            views = [question, *question_facets(question)]
            vocabulary = terms(question)
            questions[row['qid']] = []
            for view in views:
                text = f'Instruct: {INSTRUCTION}\nQuery:{view}'
                questions[row['qid']].append(items.setdefault(text, len(items)))
            for doc in row['docs']:
                text = evidence(docs[doc], vocabulary, 'dispersed')
                spans[row['qid'], doc] = items.setdefault(text, len(items))
        encoded = encode(list(items), model, tokenizer, 'cuda', args.batch_size, 1024, 'last', domain)
        for row in group:
            qid = row['qid']
            representations = encoded[questions[qid]]
            similarity = {d:(representations @ encoded[spans[qid,d]]).tolist() for d in row['docs']}
            result.append({k:row[k] for k in ('qid','domain','cohort','split')} | {'scores':similarity})
        print(domain, len(result), round(perf_counter()-started), flush=True)
    return {'model':args.model.name,'revision':REVISION,'instruction':INSTRUCTION,'per_query':result}


def evidence_views(values):
    full, *clauses = np.array(values).T
    overall = z(full)
    best = z(np.stack([z(col) for col in clauses]).max(axis=0)) if clauses else overall
    return overall, .5*(overall+best)


def replay(args):
    shards = [load(p) for p in args.scores]
    if any(s['revision']!=REVISION or s['instruction']!=INSTRUCTION for s in shards):
        raise ValueError('The snapshots differ from the pinned scoring protocol')
    scores = index({'per_query':[row for shard in shards for row in shard['per_query']]})
    return replay_views(args, scores, ARMS, evidence_views)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    scoring, analysis = (modes.add_parser(name) for name in ('score','evaluate'))
    for sub in (scoring,analysis):
        for key in ('manifest','data-root','output'):
            sub.add_argument(f'--{key}',type=Path,required=True)
    scoring.add_argument('--model',type=Path,required=True)
    for key,value in (('batch-size',24),('shards',1),('shard-index',0)):
        scoring.add_argument(f'--{key}',type=int,default=value)
    for key in ('consensus','reference','selection'):
        analysis.add_argument(f'--{key}',type=Path,required=key!='selection')
    for key in ('evidence','scores'):
        analysis.add_argument(f'--{key}',type=Path,required=True,nargs='+')
    analysis.add_argument('--confirmation',action='store_true')
    analysis.add_argument('--reserve-ids', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Provide a new output path')
    if args.mode=='score' and not (0<=args.shard_index<args.shards and args.batch_size>0):
        parser.error('Check shard and batch bounds')
    if args.mode=='evaluate' and args.confirmation and not args.selection:
        parser.error('Confirmation requires a frozen selection')
    result = score(args) if args.mode=='score' else evaluate(args,replay,ARMS)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(result,file,separators=(',',':'))


if __name__=='__main__':
    main()
