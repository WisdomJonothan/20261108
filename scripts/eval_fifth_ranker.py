"""Compare a distinct cross-encoder on the maintained system's candidate union."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

from audit_rejected_swap_evidence import read_documents
from audit_wide_vote_contributions import member
from benchmark_pairwise_replacement_v2 import strip_html, terms
from eval_event_date_windows import write_gz
from replay_anchor_cap import z
from replay_conditional_rank_fusion import PANELS, SOURCES, key, labels, measure, rank_feature, read_rows
from replay_evidence_ranking_consensus import reciprocal
from replay_incumbent_evidence import effect, macro
from score_specialist_shortlists import evidence


CANDIDATES = ('fifth_vote', 'inside_ten', 'head_agreement', 'tail_agreement')
ARMS = ('duplicate_pointwise', *CANDIDATES)


def query_text(path, qids):
    with path.open(encoding='utf-8') as stream:
        return {row['id']: ' '.join(strip_html(row['query']).split()) for line in stream
                for row in (json.loads(line),) if row['id'] in qids}


def official_inputs(archive_dir):
    archive = archive_dir / 'locked-local-vote-official-20261126.tar.gz'
    votes = {key(r): r['rankings'] for r in member(archive, 'consensus.json')['per_query']}
    base = member(archive, 'result.json')['per_query']
    if len(base) != 519 or set(votes) != {key(r) for r in base}:
        raise ValueError('Missing official vote rows')
    output = []
    for row in base:
        if row['split'] != 'dev':
            raise ValueError('Expected only the official development queries')
        current = row['rankings']['current']
        sources = [votes[key(row)][name] for name in SOURCES]
        docs = list(dict.fromkeys([*current, *(d for order in sources for d in order)]))
        output.append({'domain': row['domain'], 'qid': row['qid'], 'panel': 'official-519',
                       'docs': docs, 'current': current, 'sources': sources,
                       'prior_score': row['scores']['current']})
    return output


def same_ten_inputs(archive_dir):
    archive = archive_dir / 'paired-membership-20261127.tar.gz'
    source = member(archive, 'paired-membership-641-final-20261127.json')
    rows = [r for r in source['per_query'] if r['panel'] == 'reserve-191']
    if len(rows) != 191:
        raise ValueError('Expected all 191 reserve topics')
    return [{k: r[k] for k in ('domain', 'qid', 'panel')} |
            {'current': r['rankings']['current'], 'docs': r['rankings']['current'], 'sources': [],
             'prior_score': r['scores']['current']} for r in rows]


def prepare(args, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain']].append(row)
    output = []
    for domain, group in sorted(groups.items()):
        data = args.data_root / domain
        split = 'dev' if args.official else 'train'
        questions = query_text(data / f'examples_{split}.jsonl', {r['qid'] for r in group})
        corpus = read_documents(data / 'documents.jsonl', {d for r in group for d in r['docs']})
        for row in group:
            query = questions[row['qid']]
            words = terms(query)
            output.append({k: row[k] for k in ('domain', 'qid', 'panel', 'docs', 'current',
                                               'sources', 'prior_score')} |
                          {'query': query, 'excerpts': [evidence(corpus[d], words, 'dispersed')
                                                       for d in row['docs']]})
        print(domain, 'questions prepared', len(output), '/', len(rows), flush=True)
    return {'queries': len(rows), 'per_query': output}


def score(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with gzip.open(args.inputs, 'rt', encoding='utf-8') as stream:
        rows = json.load(stream)['per_query']
    if args.inside_only:
        for row in rows:
            if row['docs'][:10] != row['current']:
                raise ValueError('Expected maintained documents to come first in the pool')
            row['docs'] = row['docs'][:10]
            row['excerpts'] = row['excerpts'][:10]
    torch.set_num_threads(3)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16).to('cuda').eval()
    earlier = {}
    if args.head_tail_from:
        with gzip.open(args.head_tail_from, 'rt', encoding='utf-8') as stream:
            reference = json.load(stream)
        earlier = {key(row): row for row in reference['per_query']}
        if reference['model'] != args.model.name or earlier.keys() != {key(row) for row in rows}:
            raise ValueError('Matched head-only scores require the same checkpoint and population')
    pairs, active = [], []
    for row in rows:
        query, excerpts = row.pop('query'), row.pop('excerpts')
        if earlier:
            prior = earlier[key(row)]
            if row['docs'] != prior['docs']:
                raise ValueError(f'Mismatched candidate order for {key(row)}')
            tokens = tokenizer.tokenize(query, verbose=False)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            if len(ids) <= 192:
                row['new_scores'] = prior['new_scores']
                continue
            limit = ids[:64] + ids[-128:]
        else:
            limit = tokenizer.encode(query, add_special_tokens=False, truncation=True, max_length=192)
        question = tokenizer.decode(limit, skip_special_tokens=True)
        pairs.extend((question, text) for text in excerpts)
        active.append(row)
    print('pairs:', len(pairs), flush=True)
    values = []
    begun = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start+args.batch_size]
            x = tokenizer([q for q, _ in batch], [d for _, d in batch], padding=True,
                          truncation='only_second', max_length=512, return_tensors='pt').to('cuda')
            logits = model(**x).logits
            if logits.shape != (len(batch), 1):
                raise ValueError('Expected a scalar relevance logit')
            values.extend(logits[:, 0].float().tolist())
            if start % (args.batch_size * 40) == 0:
                print(len(values), '/', len(pairs), 'seconds', round(time.perf_counter()-begun), flush=True)
    iterator = iter(values)
    for row in active:
        row['new_scores'] = [next(iterator) for _ in row['docs']]
    if next(iterator, None) is not None:
        raise ValueError('Unexpected extra scores')
    return {'model': args.model.name, 'queries': len(rows), 'per_query': rows,
            'truncation': 'first64-last128' if earlier else 'first192', 'rescored_queries': len(active),
            'inference_seconds': time.perf_counter()-begun}


def order_inside(current, values):
    blend = 3*z(rank_feature(current, current)) + z([values[d] for d in current])
    return [current[i] for i in sorted(range(10), key=lambda i: -blend[i])]


def rankings(row):
    docs, current, sources = (row[k] for k in ('docs', 'current', 'sources'))
    values = dict(zip(docs, row['new_scores'], strict=True))
    new_order = sorted(docs, key=lambda doc: -values[doc])
    pointwise = sources[3]
    result = {'current': current}
    for name, vote in (('duplicate_pointwise', pointwise), ('fifth_vote', new_order[:10])):
        fused = reciprocal((*sources, vote), 60)
        result[name] = [current[0], *[doc for doc in fused if doc != current[0]][:9]]
    result['inside_ten'] = order_inside(current, values)
    head = next((d for d in new_order[:3] if d in pointwise[:5] and d in current[1:5]
                 and values[d] > values[current[0]]), None)
    upgraded_head = current.copy()
    if head is not None:
        index = current.index(head)
        upgraded_head[0], upgraded_head[index] = upgraded_head[index], upgraded_head[0]
    result['head_agreement'] = upgraded_head
    tail_pos = pointwise.index(current[9]) if current[9] in pointwise else len(pointwise)
    tail = next((d for d in new_order[:10] if d not in current and d in pointwise[:tail_pos]
                 and values[d] > values[current[9]]), None)
    result['tail_agreement'] = current[:9] + [tail] if tail is not None else current
    return result


def evaluate(args):
    with gzip.open(args.scores, 'rt', encoding='utf-8') as stream:
        scored = json.load(stream)
    selection = None
    if args.selection:
        with gzip.open(args.selection, 'rt', encoding='utf-8') as stream:
            prior = json.load(stream)
        selection = prior['selected']
        if selection not in CANDIDATES or prior['model'] != scored['model']:
            raise ValueError('The score cache needs the same model and an eligible selection')
    if args.official and (not selection or not prior['eligible'] or prior['queries'] not in (242, 433)):
        raise ValueError('The official replay requires passing transfer evidence')
    arms = (selection,) if selection else ARMS
    raw = scored['per_query']
    if args.official:
        expected, count = {'official-519'}, 519
    elif args.reserve_191:
        if selection != 'inside_ten':
            raise ValueError('The 191-topic replication only supports the locked same-ten correction')
        expected, count = {'reserve-191'}, 191
    elif selection:
        expected, count = {'reserve-242'}, 242
    else:
        expected, count = set(PANELS)-{'reserve-242'}, 450
    if {r['panel'] for r in raw} != expected or len(raw) != count:
        raise ValueError('Unexpected topic population for the requested stage')
    frozen = []
    for row in raw:
        if selection == 'inside_ten':
            values = dict(zip(row['docs'], row['new_scores'], strict=True))
            result = {'current': row['current'], selection: order_inside(row['current'], values)}
        else:
            result = rankings(row)
        frozen.append({k: row[k] for k in ('domain', 'qid', 'panel', 'prior_score')} |
                      {'rankings': {a: result[a] for a in ('current', *arms)}})
    fingerprint = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    rows = measure(frozen, labels(frozen, args.data_root, 'dev' if args.official else 'train'))
    contrasts = {arm: effect(rows, arm, 'current') for arm in arms}
    if selection:
        delta = contrasts[selection]
        eligible = min(delta['point'], delta['minimum_leave_one_domain_out']) > 0
    else:
        for arm in arms:
            contrasts[arm]['panel_omissions'] = {
                p: macro(skip, arm) - macro(skip, 'current')
                for p in sorted(expected) if (skip := [r for r in rows if r['panel'] != p])
            }
        contenders = [arm for arm in CANDIDATES
                      if min(contrasts[arm]['point'], contrasts[arm]['minimum_leave_one_domain_out'],
                             *contrasts[arm]['panel_omissions'].values()) > 0]
        if 'fifth_vote' in contenders and macro(rows, 'fifth_vote') <= macro(rows, 'duplicate_pointwise'):
            contenders.remove('fifth_vote')
        selection = max(contenders, key=lambda a: contrasts[a]['point'], default=None)
        eligible = selection is not None
    return scored | {'selected': selection, 'eligible': eligible, 'rank_fingerprint': fingerprint,
                     'scores': {a: macro(rows, a) for a in ('current', *arms)},
                     'contrasts': contrasts, 'evaluated_rankings': rows}


def combined_transfer(args):
    parts = []
    for path in (args.transfer_242, args.transfer_191):
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            parts.append(json.load(stream))
    if ([p['queries'] for p in parts] != [242, 191] or parts[0]['model'] != parts[1]['model']
            or any(p['selected'] != 'inside_ten' or p['truncation'] != 'first192' for p in parts)):
        raise ValueError('Expected the unchanged rule on both transfer populations')
    rows = [r for part in parts for r in part['evaluated_rankings']]
    if len({key(r) for r in rows}) != 433:
        raise ValueError('The transfer populations must be disjoint')
    delta = effect(rows, 'inside_ten', 'current')
    eligible = parts[1]['eligible'] and min(delta['point'], delta['minimum_leave_one_domain_out']) > 0
    return {'selected': 'inside_ten', 'queries': len(rows), 'model': parts[0]['model'],
            'eligible': eligible, 'scores': {a: macro(rows, a) for a in ('current', 'inside_ten')},
            'contrasts': {'inside_ten': delta},
            'rank_fingerprints': [p['rank_fingerprint'] for p in parts], 'evaluated_rankings': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--archive-dir', type=Path, required=True)
    population = prep.add_mutually_exclusive_group()
    population.add_argument('--confirmation', action='store_true')
    population.add_argument('--official', action='store_true')
    population.add_argument('--reserve-191', action='store_true')
    run = commands.add_parser('score')
    run.add_argument('--inputs', type=Path, required=True)
    run.add_argument('--model', type=Path, required=True)
    run.add_argument('--batch-size', type=int, default=16)
    run.add_argument('--head-tail-from', type=Path)
    run.add_argument('--inside-only', action='store_true')
    metric = commands.add_parser('evaluate')
    metric.add_argument('--scores', type=Path, required=True)
    metric.add_argument('--selection', type=Path)
    metric.add_argument('--official', action='store_true')
    metric.add_argument('--reserve-191', action='store_true')
    combine = commands.add_parser('combine-transfer')
    combine.add_argument('--transfer-242', type=Path, required=True)
    combine.add_argument('--transfer-191', type=Path, required=True)
    for command in (prep, metric):
        command.add_argument('--data-root', type=Path, required=True)
    for command in (prep, run, metric, combine):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    if args.stage == 'prepare':
        panels = ('reserve-242',) if args.confirmation else tuple(p for p in PANELS if p != 'reserve-242')
        if args.official:
            rows = official_inputs(args.archive_dir)
        elif args.reserve_191:
            rows = same_ten_inputs(args.archive_dir)
        else:
            rows = read_rows(args.archive_dir, panels)
        result = prepare(args, rows)
    elif args.stage == 'score':
        result = score(args)
    elif args.stage == 'evaluate':
        result = evaluate(args)
    else:
        result = combined_transfer(args)
    write_gz(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_query', 'evaluated_rankings')}, indent=2),
          flush=True)


if __name__ == '__main__':
    main()
