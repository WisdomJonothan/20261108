"""Measure explicit temporal contradictions separately from missing evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean
import time

import numpy as np

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_qwen_reranker_adapter import choose, grade, summarize
from replay_anchor_cap import z
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from score_specialist_shortlists import evidence


RUBRIC = (
    'Classify this document as evidence for the question. Choose one category. '
    '0: unrelated to the question. '
    '1: on topic, but missing evidence for any specific part of the question. '
    '2: appears to answer a specific part, but the claim is explicitly about the wrong '
    'time, order of events, or version for that part. A missing date by itself belongs in 1. '
    '3: offers usable evidence for at least one specific part and respects the '
    'time, order, and version conditions for that part. A later source can describe an earlier '
    'event. Evaluate the time of the claim itself. A matching date by itself is not evidence. '
    'If the document gives usable evidence for one part, select 3 even if other parts go unanswered. '
    'Output only the category digit.'
)
FACET_RUBRIC = (
    'Check whether the document provides usable evidence for a concrete part of the full question. '
    'Respect the requested time, event order, version and entities. Use the neighboring context to '
    'interpret each numbered request. Give the number of the listed request most clearly answered. '
    'Give 5 if the document gives evidence for a different part of the full question. '
    'Give 0 if it provides no usable evidence. Output only the digit.'
)
VOTERS = ('current', 'reasoned', 'verified', 'pointwise')
POLICIES = ('reference', 'direct14', 'direct22', 'temporal14', 'temporal22',
            'demote_conflict', 'replace_conflict')


def question_facets(query):
    facets, seen = [], []
    for sentence in query.split('?')[:-1]:
        text = re.split(r'(?<=[.!;])\s+', sentence.strip())[-1].strip()
        vocabulary = terms(text)
        if len(vocabulary) < 3 or any(len(old & vocabulary)/len(old | vocabulary) > .8 for old in seen):
            continue
        facets.append(text + '?')
        seen.append(vocabulary)
    return facets[:4]


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = load(args.manifest)['queries'][args.shard_index::args.shards]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left', local_files_only=True)
    labels = '012345' if args.facets else '0123'
    tokens = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(ids) != 1 for ids in tokens) or len({ids[0] for ids in tokens}) != len(labels):
        raise ValueError('Expected distinct one-token categories')
    ids = [item[0] for item in tokens]
    groups = defaultdict(list)
    for row in source:
        groups[row['domain']].append(row)
    rows, started = [], time.perf_counter()
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            texts = read_documents(args.data_root / domain / 'documents.jsonl',
                                   {d for query in group for d in query['docs']})
            inputs = []
            for row in group:
                query = row['query']
                vocabulary = terms(query)
                rubric = RUBRIC
                if args.facets:
                    parts = '\n'.join(f'{i}. {text}' for i, text in enumerate(question_facets(query), 1))
                    rubric = f'{FACET_RUBRIC}\n\nListed requests\n{parts}'
                for doc in row['docs']:
                    excerpt = evidence(texts[doc], vocabulary, 'dispersed')
                    content = f'{rubric}\n\nQuestion\n{query}\n\nDocument\n{excerpt}'
                    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                          tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False)
                    inputs.append(prompt + 'Category: ')
            logits = []
            for start in range(0, len(inputs), args.batch_size):
                batch = tokenizer(inputs[start:start + args.batch_size], padding=True, add_special_tokens=False,
                                  return_tensors='pt').to(model.device)
                if batch.input_ids.shape[-1] > 4096:
                    raise ValueError(f'Text exceeds the evidence window in {domain}')
                values = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, ids]
                logits.extend(values.float().cpu().tolist())
            count = 0
            for row in group:
                docs = row['docs']
                end = count + len(docs)
                values = dict(zip(docs, logits[count:end], strict=True))
                item = {k: row[k] for k in ('domain', 'qid', 'split', 'cohort')} | {'categories': values}
                if args.facets:
                    item['facet_count'] = len(question_facets(row['query']))
                rows.append(item)
                count = end
            if count != len(logits):
                raise ValueError(f'Unassigned category scores in {domain}')
            print(domain, len(rows), '/', len(source), round(time.perf_counter() - started), flush=True)
    return {'model': args.model.name, 'rubric': FACET_RUBRIC if args.facets else RUBRIC, 'per_query': rows,
            'inference': {'seconds': time.perf_counter()-started,
                          'pairs': sum(len(r['categories']) for r in rows),
                          'peak_mib': torch.cuda.max_memory_allocated()/2**20}}


def ranking(docs, values):
    blended = z(1 / np.log2(2 + np.arange(len(docs)))) + z(values)
    return [docs[i] for i in np.argsort(-blended, kind='stable')[:10]]


def conflict_action(baseline, docs, category, conflict, allow_new):
    eligible = [(i, d) for i, d in enumerate(baseline) if conflict.get(d, 0.) >= 1.]
    if not eligible:
        return baseline
    position, target = max(eligible, key=lambda item: conflict[item[1]])
    grade = dict(zip(docs, category[:, 3]-np.logaddexp.reduce(category[:, :3], axis=1), strict=True))
    challengers = [d for d in (docs if allow_new else baseline[position+1:])
                   if d in grade and (d not in baseline if allow_new else True)]
    revised = baseline.copy()
    if challengers:
        contender = max(challengers, key=grade.__getitem__)
        if grade[contender] >= .5 and grade[contender] > grade[target]:
            if allow_new:
                revised[position] = contender
            else:
                revised.remove(contender)
                revised.insert(position, contender)
                revised.remove(target)
                revised.append(target)
    return revised


def prepare(args):
    queries = index({'per_query': load(args.manifest)['queries']})
    base, heads = index(load(args.consensus)), index(load(args.head))
    cached = index({'per_query': [row for f in args.original_scores for row in load(f)['per_query']]})
    shards = [load(path) for path in args.categories]
    if len({(s['model'], s['rubric']) for s in shards}) != 1:
        raise ValueError('Every prediction shard must use the same checker')
    category = index({'per_query': [r for shard in shards for r in shard['per_query']]})
    if not all(q.keys() == queries.keys() for q in (base, heads, cached, category)):
        raise ValueError('Expected the same question IDs in cached model scores and rankings')

    rows = []
    for key, r in queries.items():
        docs = r['docs']
        values = category[key]['categories']
        obs = cached[key]['observations']
        if len(set(docs)) != 22 or set(docs) != values.keys() or set(docs[:14]) != obs.keys():
            raise ValueError(f'Candidate membership differs for {key}')
        ranks = base[key]['rankings']
        votes = tuple(ranks[a] for a in VOTERS)
        prior = list(heads[key]['rankings']['verified_0.5'])
        if head(reciprocal(votes, 60), ranks['pointwise'], obs, True, .5) != prior:
            raise ValueError(f'Frozen consensus and head decision differ for {key}')
        matrix = np.array([values[d] for d in docs])
        direct = matrix[:, 3] - np.logaddexp.reduce(matrix[:, :3], axis=1)
        conflict = matrix[:, 2] - matrix[:, [0, 1, 3]].max(axis=1)
        temporal = direct - np.maximum(0, conflict)
        orders = {'current': prior, 'reference': prior,
                  'previous_reference': heads[key]['rankings']['current']}
        for name, score_values in (('direct', direct), ('temporal', temporal)):
            for width in (14, 22):
                new_vote = ranking(docs[:width], score_values[:width])
                combine = reciprocal((*votes, new_vote), 60)
                orders[f'{name}{width}'] = head(combine, ranks['pointwise'], obs, True, .5)
        conflicts = dict(zip(docs, conflict, strict=True))
        orders['demote_conflict'] = conflict_action(prior, docs, matrix, conflicts, False)
        orders['replace_conflict'] = conflict_action(prior, docs, matrix, conflicts, True)
        rows.append({k: r[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                    {'rankings': orders, 'strong_conflict_count': int((conflict >= 1).sum())})
    return rows


def evaluate(args):
    split = 'dev' if args.selection and not args.confirmation else 'train'
    rows = grade(prepare(args), split, args.data_root)
    expected = 155 if args.confirmation else (519 if split == 'dev' else 186)
    if len(rows) != expected:
        raise ValueError('Unexpected evaluation population')
    if args.selection:
        selected = load(args.selection)['selected']
        if selected not in POLICIES:
            raise ValueError('Unknown locked choice')
        summary = summarize(rows, selected)
        if split == 'dev' and abs(summary['reference']['score'] - .40674178096988606) > 1e-12:
            raise ValueError('Official baseline differs from the frozen version')
        report = {'confirmation' if args.confirmation else 'dev': summary}
    else:
        selected = choose(rows, POLICIES)
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], POLICIES)
                   for f in sorted({r['cohort'] for r in rows})}
        summary = summarize(rows, selected, by_fold)
        report = {'training': summary, 'by_fold': by_fold,
                  'original_65': summarize([r for r in rows if r['actual_reference']], selected, by_fold)}
    report.update({'selected': selected, 'per_query': rows, 'policies': POLICIES})
    print(selected, {a: round(v['score'], 8) for a, v in summary.items()}, flush=True)
    delta = summary['selected']['change']['per_domain']
    smallest = min(mean(v for d, v in delta.items() if d != omit) for omit in delta)
    print('Minimum leave-one-domain-out change', smallest, flush=True)
    report['selected_minimum_leave_one_domain_out'] = smallest
    for width in (14, 22):
        direct, temporal = f'direct{width}', f'temporal{width}'
        report[f'contrast_at_{width}'] = paired_delta(
            [(r['domain'], r['scores'][temporal]-r['scores'][direct]) for r in rows])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest='phase', required=True)
    scoring, evaluation = (phases.add_parser(phase) for phase in ('score', 'evaluate'))
    for phase in (scoring, evaluation):
        for name in ('manifest', 'data-root', 'output'):
            phase.add_argument(f'--{name}', type=Path, required=True)
    scoring.add_argument('--model', type=Path, required=True)
    scoring.add_argument('--batch-size', type=int, default=12)
    scoring.add_argument('--facets', action='store_true')
    scoring.add_argument('--shards', type=int, default=1)
    scoring.add_argument('--shard-index', type=int, default=0)
    for name in ('consensus', 'head', 'selection'):
        evaluation.add_argument(f'--{name}', type=Path, required=name != 'selection')
    evaluation.add_argument('--confirmation', action='store_true')
    for name in ('original-scores', 'categories'):
        evaluation.add_argument(f'--{name}', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.phase == 'score' and not 0 <= args.shard_index < args.shards:
        parser.error('The shard index must lie within the chosen number of shards')
    if args.phase == 'evaluate' and args.confirmation and args.selection is None:
        parser.error('Confirm the previously locked training choice')
    result = (score if args.phase == 'score' else evaluate)(args)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
