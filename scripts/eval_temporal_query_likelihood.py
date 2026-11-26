"""Contrast whole-question and clause query likelihood on the same evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_temporal_condition_intervention import evaluate, replay_views
from eval_temporal_evidence_categories import question_facets
from replay_anchor_cap import z
from replay_anchor_fallback import index
from score_specialist_shortlists import evidence


PROMPT = 'Write the question that this source evidence could help answer.\n\nSource evidence\n'
TEMPORAL = re.compile(
    r'\b(?:1[0-9]{3}|20[0-9]{2}|before|after|since|until|during|throughout|'
    r'precedes|followed|earlier|later|century|decade|version|release)\b|\bv?\d+\.\d+(?:\.\d+)?\b',
    re.IGNORECASE,
)
ARMS = ('current', 'whole_question', 'best_clause', 'best_clause_with_time')


def mask_tokens(offsets, ranges):
    return [i for i, (start, end) in enumerate(offsets) if end>start and
            any(start < right and end > left for left, right in ranges)]


def partitions(question, offsets):
    fragments = []
    for clause in question_facets(question):
        start = question.find(clause)
        if start >= 0:
            subset = mask_tokens(offsets, [(start, start+len(clause))])
            if subset:
                fragments.append(subset)
    temporal = mask_tokens(offsets, [m.span() for m in TEMPORAL.finditer(question)])
    return fragments, temporal


def score_question(row, documents, model, tokenizer, requested_batch):
    import torch
    from torch.nn.functional import cross_entropy

    question = ' ' + row['query'].strip()
    encoded = tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    if not ids:
        raise ValueError(f'Missing target tokens for {row["qid"]}')
    groups, time = partitions(question, encoded['offset_mapping'])
    vocabulary = terms(question)
    candidates = row['docs']
    width = max(1, min(requested_batch, 1152//len(ids)))
    output = {}
    target = torch.tensor(ids, device=model.device)
    for start in range(0,len(candidates),width):
        selected = candidates[start:start+width]
        prompts = []
        for doc in selected:
            excerpt = evidence(documents[doc], vocabulary, 'dispersed')
            chat = tokenizer.apply_chat_template([{'role':'user', 'content':PROMPT+excerpt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(chat)
        prefix = tokenizer(prompts, add_special_tokens=False)['input_ids']
        tokens = tokenizer.pad({'input_ids': [text+ids for text in prefix]}, padding=True,
                               return_tensors='pt').to(model.device)
        if tokens.input_ids.shape[1] > 4096:
            raise ValueError(f'Query and evidence exceed scoring scope for {row["qid"]}')
        logits = model(**tokens, use_cache=False, logits_to_keep=len(ids)+1).logits[:, :-1, :]
        logp = -cross_entropy(logits.float().flatten(0, 1), target.repeat(len(selected)),
                             reduction='none').reshape(len(selected), -1).cpu().numpy()
        for doc, values in zip(selected, logp, strict=True):
            result = [float(values.mean()), *[float(values[index].mean()) for index in groups]]
            if time:
                result.append(float(values[time].mean()))
                has_time = True
            else:
                result.append(result[0])
                has_time = False
            output[doc] = result
    return output, has_time


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load(args.manifest)['queries'][args.shard_index::args.shards]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', local_files_only=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain']].append(row)
    started = perf_counter()
    result = []
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            docs = read_documents(args.data_root/domain/'documents.jsonl',
                                  {doc for row in group for doc in row['docs']})
            for row in group:
                likelihoods, has_time = score_question(row, docs, model, tokenizer, args.batch_size)
                result.append({k:row[k] for k in ('domain','qid','split','cohort')} |
                              {'has_time':has_time, 'scores':likelihoods})
            print(domain, len(result), '/', len(rows), round(perf_counter()-started), flush=True)
    return {'model':args.model.name, 'prompt':PROMPT, 'per_query':result}


def likelihood_views(values):
    whole, *clauses, temporal = np.array(values).T
    overall = z(whole)
    best = z(np.stack([z(col) for col in clauses]).max(axis=0)) if clauses else overall
    focused = .5*(overall+best)
    return overall, focused, .75*focused+.25*z(temporal)


def replay(args):
    parts = [load(p) for p in args.scores]
    if len({p['model'] for p in parts}) != 1 or any(p['prompt'] != PROMPT for p in parts):
        raise ValueError('The likelihood shards must share the same model and prompt')
    scores = index({'per_query':[row for part in parts for row in part['per_query']]})
    return replay_views(args, scores, ARMS, likelihood_views)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    scoring, analysis = (commands.add_parser(name) for name in ('score','evaluate'))
    for step in (scoring,analysis):
        for key in ('manifest','data-root','output'):
            step.add_argument(f'--{key}',type=Path,required=True)
    scoring.add_argument('--model',type=Path,required=True)
    for key, default in (('batch-size',14),('shards',1),('shard-index',0)):
        scoring.add_argument(f'--{key}',type=int,default=default)
    for key in ('consensus','reference','selection'):
        analysis.add_argument(f'--{key}',type=Path,required=key!='selection')
    for key in ('evidence','scores'):
        analysis.add_argument(f'--{key}',type=Path,nargs='+',required=True)
    analysis.add_argument('--confirmation',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.command=='score' and not (0<=args.shard_index<args.shards and args.batch_size>0):
        parser.error('Check shard and batch bounds')
    if args.command=='evaluate' and args.confirmation and not args.selection:
        parser.error('Confirmation requires an already frozen choice')
    result = score(args) if args.command=='score' else evaluate(args,replay,ARMS)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(result,file,separators=(',',':'))


if __name__=='__main__':
    main()
