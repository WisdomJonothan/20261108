"""Measure whether disregarding temporal restrictions changes evidence ranking."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_qwen_reranker_adapter import choose, grade, summarize
from replay_anchor_cap import z
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from score_specialist_shortlists import evidence


BASIC = (
    'Judge if the document gives usable evidence for at least one specific request. '
    'Apply the entities and substantive requirements of that request. '
    'A document can answer one request and leave other requests unanswered. '
)
CONDITIONS = {
    'full': 'Apply the time period, ordering and version constraints of that request.',
    'neutral': 'Set aside the time period, ordering and version constraints of that request.',
}
VOTERS = ('current', 'reasoned', 'verified', 'pointwise')
ARMS = ('current', 'full_vote', 'neutral_vote', 'conditioned_vote', 'contrast_vote')


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load(args.manifest)['queries'][args.shard_index::args.shards]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                               local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', local_files_only=True)
    ids = [tokenizer.encode(answer, add_special_tokens=False) for answer in ('yes', 'no')]
    if any(len(token) != 1 for token in ids) or ids[0] == ids[1]:
        raise ValueError('The checkpoint must have distinct one-token yes and no labels')
    ids = [value[0] for value in ids]
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain']].append(row)
    output = []
    started = perf_counter()
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            docs = read_documents(args.data_root / domain / 'documents.jsonl',
                                  {doc for row in group for doc in row['docs']})
            prompts = []
            for row in group:
                question = row['query']
                vocabulary = terms(question)
                for doc in row['docs']:
                    excerpt = evidence(docs[doc], vocabulary, 'dispersed')
                    message = f'Question\n{question}\n\nDocument\n{excerpt}'
                    for condition in CONDITIONS.values():
                        text = f'{BASIC}{condition} Answer only yes or no.\n\n{message}'
                        chat = tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                            tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        prompts.append(chat + 'Answer: ')
            scores = []
            for offset in range(0, len(prompts), args.batch_size):
                batch = tokenizer(prompts[offset:offset+args.batch_size], padding=True,
                                  add_special_tokens=False, return_tensors='pt').to('cuda')
                if batch.input_ids.size(1) > 4096:
                    raise ValueError(f'Expected a bounded evidence window in {domain}')
                values = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float()
                scores.extend((values[:, 0] - values[:, 1]).cpu().tolist())
            cursor = 0
            for row in group:
                end = cursor + len(row['docs']) * len(CONDITIONS)
                pairs = np.array(scores[cursor:end]).reshape(-1, len(CONDITIONS)).tolist()
                output.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'split')} |
                              {'scores': dict(zip(row['docs'], pairs, strict=True))})
                cursor = end
            if cursor != len(scores):
                raise ValueError(f'Unassigned model scores in {domain}')
            print(domain, len(output), '/', len(rows), round(perf_counter()-started), flush=True)
    return {'model': args.model.name, 'conditions': CONDITIONS, 'per_query': output}


def replay_views(args, scores, arms, views):
    queries = index({'per_query': load(args.manifest)['queries']})
    consensuses, references = (index(load(p)) for p in (args.consensus, args.reference))
    observations = index({'per_query': [r for p in args.evidence for r in load(p)['per_query']]})
    if not all(source.keys() == queries.keys() for source in (references, consensuses, scores, observations)):
        raise ValueError('The replay requires the same query IDs from every source')
    results = []
    for key, row in queries.items():
        docs = row['docs']
        values = scores[key]['scores']
        voters = consensuses[key]['rankings']
        cached = observations[key]['observations']
        if len(docs) != 14 or set(docs) != values.keys() or values.keys() != cached.keys():
            raise ValueError(f'Mismatched shortlist membership for {key}')
        vote_lists = tuple(voters[arm] for arm in VOTERS)
        prior = references[key]['rankings']['verified_0.5']

        def combine(lists):
            return head(reciprocal(lists, 60), voters['pointwise'], cached, True, .5)

        if combine(vote_lists) != prior:
            raise ValueError(f'Unexpected baseline ranking for {key}')
        document_rank = z(1/np.log2(2+np.arange(len(docs))))
        orders = {'current': prior}
        for arm, view in zip(arms[1:], views([values[d] for d in docs]), strict=True):
            order = np.argsort(-(document_rank + view), kind='stable')[:10]
            orders[arm] = combine((*vote_lists, [docs[i] for i in order]))
        results.append({k: row[k] for k in ('domain','qid','cohort','split','actual_reference')} |
                       {'rankings': orders})
    return results


def condition_views(values):
    full, neutral = np.array(values).T
    # A positive difference means that applying the condition helped the same
    # source. A negative difference means the broader notion of topic helped.
    residual = z(full-neutral)
    return z(full), z(neutral), (z(full)+residual)/2, residual


def replay(args):
    shards = [load(p) for p in args.scores]
    if len({part['model'] for part in shards}) != 1 or any(part['conditions'] != CONDITIONS for part in shards):
        raise ValueError('The shards use different scoring configurations')
    scores = index({'per_query': [r for part in shards for r in part['per_query']]})
    return replay_views(args, scores, ARMS, condition_views)


def evaluate(args, replay=replay, arms=ARMS):
    reserve = getattr(args, 'reserve_ids', None)
    split = 'dev' if args.selection and not args.confirmation else 'train'
    candidate_rows = replay(args)
    if reserve:
        expected = {tuple(key) for key in load(reserve)['queries']}
        observed = {(r['domain'], r['qid']) for r in candidate_rows}
        if not(args.selection and args.confirmation) or observed != expected or len(candidate_rows) != len(expected):
            raise ValueError('The reserve must receive a frozen policy and exactly its recorded query IDs')
    elif len(candidate_rows) != (155 if args.confirmation else (519 if split == 'dev' else 186)):
        raise ValueError('The evaluation population differs from the planned split')
    selected = load(args.selection)['selected'] if args.selection else None
    if selected and selected not in arms:
        raise ValueError('The locked policy does not exist in the planned comparison')
    if reserve or split == 'dev':
        candidate_rows = [row | {'rankings': {arm: row['rankings'][arm]
                                              for arm in dict.fromkeys(('current', selected))}}
                          for row in candidate_rows]
    rows = grade(candidate_rows, split, args.data_root)
    if selected:
        changes = summarize(rows, selected)
        if split == 'dev' and abs(changes['current']['score']-.40674178096988606)>1e-12:
            raise ValueError('The official-dev reference differs from the frozen ranking')
        result = {'confirmation' if args.confirmation else 'dev': changes}
    else:
        selected = choose(rows, arms)
        selections = {f: choose([r for r in rows if r['cohort']!=f], arms)
                      for f in sorted({r['cohort'] for r in rows})}
        changes = summarize(rows, selected, selections)
        result = {'training': changes, 'by_fold': selections}
    domain = changes['selected']['change']['per_domain']
    result.update({'selected': selected, 'per_query': rows,
                   'reserved_ids': bool(reserve),
                   'selected_minimum_leave_one_domain_out': min(
                       mean(value for d,value in domain.items() if d!=omit) for omit in domain)})
    print(selected, {a:round(v['score'],8) for a,v in changes.items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    scorer, analysis = (commands.add_parser(name) for name in ('score','evaluate'))
    for command in (scorer,analysis):
        for path in ('manifest','data-root','output'):
            command.add_argument(f'--{path}', type=Path, required=True)
    scorer.add_argument('--model', required=True, type=Path)
    for key, default in (('batch-size',12), ('shards',1), ('shard-index',0)):
        scorer.add_argument(f'--{key}', type=int, default=default)
    for path in ('consensus','reference','selection'):
        analysis.add_argument(f'--{path}', type=Path, required=path!='selection')
    for path in ('evidence','scores'):
        analysis.add_argument(f'--{path}', type=Path, required=True, nargs='+')
    analysis.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Provide a new output path')
    if args.command=='score' and not 0<=args.shard_index<args.shards:
        parser.error('Check shard index and shard count')
    if args.command=='evaluate' and args.confirmation and not args.selection:
        parser.error('Confirmation requires a preselected policy')
    output = (score if args.command=='score' else evaluate)(args)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(output, file, separators=(',',':'))


if __name__ == '__main__':
    main()
