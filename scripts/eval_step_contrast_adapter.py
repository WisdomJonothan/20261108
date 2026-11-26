"""Evaluate the fixed-pool relevance contribution of learning which clause a source supports."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_temporal_condition_intervention import evaluate, replay_views
from prepare_step_contrast_adapter import literal_requests, pack_inputs
from replay_anchor_cap import z
from replay_anchor_fallback import index
from run_qwen3_rerank import PREFIX, SUFFIX
from score_specialist_shortlists import evidence


REVISION = '0101299a7937a016848d8d55b092f6e31cfa312d'
MODELS = ('frozen', 'full', 'contrast')
ARMS = ('current', *(f'{model}/{scope}' for model in MODELS for scope in ('whole','clauses')))
ELIGIBLE = ('current', 'contrast/clauses')


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qwen_reranker_adapter import load_adapter
    from train_qwen_reranker_adapter import JudgedScore

    source = load(args.manifest)['queries'][args.shard_index::args.shards]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                local_files_only=True).to('cuda').eval()
    identity = {'revision':REVISION, 'variant':args.variant}
    if args.adapter:
        identity['sha256'] = sha256(args.adapter.read_bytes()).hexdigest()
        identity['training'] = load_adapter(model, args.adapter)
    scorer = JudgedScore(model, tokenizer).eval()
    tokens = [tokenizer.encode(text, add_special_tokens=False) for text in (PREFIX,SUFFIX)]
    groups = defaultdict(list)
    for row in source:
        groups[row['domain']].append(row)
    results = []
    started = perf_counter()
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            texts = read_documents(args.data_root/domain/'documents.jsonl',
                                   {d for row in group for d in row['docs']})
            prompts, widths = [], []
            for row in group:
                q = row['query']
                vocab = terms(q)
                snippets = [evidence(texts[doc], vocab, 'dispersed') for doc in row['docs']]
                views = [None, *literal_requests(q)]
                widths.append(len(views))
                for view in views:
                    prompts.extend(pack_inputs(q, view, snippets, tokenizer, tokens, 1024))
            scores = []
            for i in range(0,len(prompts),args.batch_size):
                inputs = tokenizer.pad({'input_ids':prompts[i:i+args.batch_size]},padding=True,
                                       return_tensors='pt').to('cuda')
                scores.extend(scorer(**inputs).cpu().tolist())
            cursor = 0
            for row,n_views in zip(group,widths,strict=True):
                docs = row['docs']
                end = cursor + len(docs)*n_views
                views = np.array(scores[cursor:end]).reshape(n_views,len(docs)).T.tolist()
                results.append({k:row[k] for k in ('domain','qid','split','cohort')} |
                               {'scores':dict(zip(docs,views,strict=True))})
                cursor = end
            if cursor != len(scores):
                raise ValueError(f'Unexpected number of scores for {domain}')
            print(domain,len(results),round(perf_counter()-started),flush=True)
    return {'identity':identity, 'per_query':results}


def focus_views(values):
    predictions = np.array(values)
    arms = []
    for j in range(len(MODELS)):
        arms.append(whole_and_clauses(predictions[:, j, :]))
    return [view for pair in arms for view in pair]


def whole_and_clauses(matrix):
    scores = z(matrix[:,0])
    if matrix.shape[1] == 1:
        return scores, scores
    clauses = np.column_stack([z(column) for column in matrix[:,1:].T])
    return scores, .5*(scores+z(clauses.max(axis=1)))


def replay(args):
    shards = defaultdict(list)
    for path in args.scores:
        value = load(path)
        if value['identity']['revision'] != REVISION:
            raise ValueError('A score file belongs to a different base model')
        shards[value['identity']['variant']].append(value)
    cached = {}
    for model in MODELS:
        parts = shards[model]
        if not parts or any(x['identity'] != parts[0]['identity'] for x in parts):
            raise ValueError(f'Missing shards or mixed checkpoint identities for {model}')
        cached[model] = index({'per_query':[r for part in parts for r in part['per_query']]})
    baseline = cached['frozen']
    if any(x.keys()!=baseline.keys() for x in cached.values()):
        raise ValueError('Every model needs to score the same queries')
    combined = {}
    for key,row in baseline.items():
        views = [cached[model][key]['scores'] for model in MODELS]
        if any(part.keys()!=row['scores'].keys() for part in views):
            raise ValueError(f'Incompatible candidate IDs for {key}')
        combined[key] = {'scores':{d:[v[d] for v in views] for d in row['scores']}}
    return replay_views(args,combined,ARMS,focus_views)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    steps=parser.add_subparsers(dest='step',required=True)
    scoring,analysis=(steps.add_parser(stage) for stage in ('score','evaluate'))
    for stage in (scoring,analysis):
        for key in ('manifest','data-root','output'):
            stage.add_argument(f'--{key}',type=Path,required=True)
    scoring.add_argument('--model',type=Path,required=True)
    scoring.add_argument('--adapter',type=Path)
    scoring.add_argument('--variant',choices=MODELS,required=True)
    for key,default in (('batch-size',24),('shards',1),('shard-index',0)):
        scoring.add_argument(f'--{key}',type=int,default=default)
    for key in ('consensus','reference','selection','reserve-ids'):
        analysis.add_argument(f'--{key}',type=Path,required=key not in ('selection','reserve-ids'))
    for key in ('evidence','scores'):
        analysis.add_argument(f'--{key}',type=Path,required=True,nargs='+')
    analysis.add_argument('--confirmation',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    if args.step=='score' and not ((args.variant=='frozen') == (args.adapter is None)):
        parser.error('The full and contrast variants require their own adapter')
    if args.step=='score' and not (0<=args.shard_index<args.shards and args.batch_size>0):
        parser.error('Check inference shard and batch bounds')
    if args.step=='evaluate' and args.confirmation and not args.selection:
        parser.error('Confirmation requires a frozen policy')
    result=score(args) if args.step=='score' else evaluate(args,replay,ELIGIBLE)
    with args.output.open('x',encoding='utf8') as out:
        json.dump(result,out,separators=(',',':'))


if __name__=='__main__':
    main()
