"""Evaluate a reasoning encoder on a larger union of two corpus retrieval lists."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import strip_html
from compare_anchor_reference import load
from eval_diver_evidence_views import INSTRUCTION, REVISION
from lib_reteco import load_examples, read_trec_run
from replay_anchor_fallback import index
from retrieve_after_evidence_review import audit


SOURCES = ('mxbai', 'bge', 'diver')


def four_new(order, original):
    found = [d for d in order if d not in original][:4]
    if len(found) != 4:
        raise ValueError('The common source did not provide four new candidates')
    return found


def union_for(row, first, second, depth):
    qid = row['qid']
    old = row['docs']
    if len(old) != 60 or len(set(old)) != 60:
        raise ValueError(f'Expected an unchanged 60-document pool for {row["domain"]}/{qid}')
    parent_orders = tuple([d for _, _, d in run[qid][:depth]] for run in (first, second))
    union = list(dict.fromkeys(parent_orders[0] + parent_orders[1] + old))
    return parent_orders, union


def score(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from run_dense_generic import encode

    rows = load(args.manifest)['queries'][args.shard_index::args.shards]
    frozen = index(load(args.reference))
    model = AutoModel.from_pretrained(args.model,dtype=torch.bfloat16,
                                     local_files_only=True).to('cuda').eval()
    tok = AutoTokenizer.from_pretrained(args.model,padding_side='right',local_files_only=True)
    groups = defaultdict(list)
    for r in rows:
        groups[r['domain'],r['split']].append(r)
    output = []
    start = perf_counter()
    for (domain,split), group in sorted(groups.items()):
        dense, alternative = (read_trec_run(root/f'dense_{tag}_{domain}_{split}.trec')
                              for root,tag in ((args.dense_runs,'mxbai'),(args.bge_runs,'bge')))
        queries = dict(zip(*load_examples(args.data_root/domain,split),strict=True))
        parents, candidates = {}, {}
        for r in group:
            qid = r['qid']
            parents[qid], candidates[qid] = union_for(r, dense, alternative, args.depth)
        docs = list(dict.fromkeys(d for pool in candidates.values() for d in pool))
        source = read_documents(args.data_root/domain/'documents.jsonl',set(docs))
        snippets = [(strip_html(source[d][:51200]) or ' ') for d in docs]
        doc_index = {d:i for i,d in enumerate(docs)}
        vectors = encode(snippets,model,tok,'cuda',args.batch_size,512,'last',domain)
        query_vectors = encode([f'Instruct: {INSTRUCTION}\nQuery:{queries[r["qid"]]}' for r in group],
                               model,tok,'cuda',args.batch_size,512,'last',domain+' queries')
        for r,vector in zip(group,query_vectors,strict=True):
            qid = r['qid']
            old = set(r['docs'])
            matchable = candidates[qid]
            values = vectors[[doc_index[d] for d in matchable]] @ vector
            ranked = [matchable[i] for i in np.argsort(-values,kind='stable')]
            offers = dict(zip(SOURCES, (four_new(o,old) for o in (*parents[qid],ranked)),strict=True))
            output.append({k:r[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                          {'original':r['docs'],'prior_ten':frozen[domain,qid]['rankings']['verified_0.5'],
                           'offers':offers,'union_count':len(matchable)})
        print(domain,len(output),len(docs),'unique documents',round(perf_counter()-start),flush=True)
    return {'models':{'diver_revision':REVISION,'source_depth':args.depth},
            'sources':list(SOURCES),'per_query':output}


def combine(args):
    pieces = [load(p) for p in args.shards]
    if not pieces or any(x['models'] != pieces[0]['models'] or x['sources'] != pieces[0]['sources']
                         for x in pieces):
        raise ValueError('The retrieval shards have mismatched configurations')
    result = {'models':pieces[0]['models'],'sources':pieces[0]['sources'],
              'per_query':[r for p in pieces for r in p['per_query']]}
    if len(index(result))!=len(result['per_query']):
        raise ValueError('Duplicate query IDs in the search shards')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    stages=parser.add_subparsers(dest='stage',required=True)
    scorer, merger, diagnostic=(stages.add_parser(s) for s in ('score','combine','audit'))
    for stage in (scorer,merger,diagnostic):
        stage.add_argument('--output',type=Path,required=True)
    for k in ('manifest','reference','model','data-root','dense-runs','bge-runs'):
        scorer.add_argument(f'--{k}',type=Path,required=True)
    for k,default in (('depth',200),('batch-size',16),('shards',1),('shard-index',0)):
        scorer.add_argument(f'--{k}',type=int,default=default)
    merger.add_argument('--shards',type=Path,nargs='+',required=True)
    diagnostic.add_argument('--pools',type=Path,required=True)
    diagnostic.add_argument('--data-root',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.stage=='score' and not (args.depth>=64 and args.batch_size>0 and 0<=args.shard_index<args.shards):
        parser.error('Check the source depth, shard index and batch size')
    result={'score':score,'combine':combine,'audit':audit}[args.stage](args)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(result,file,separators=(',',':'))


if __name__=='__main__':
    main()
