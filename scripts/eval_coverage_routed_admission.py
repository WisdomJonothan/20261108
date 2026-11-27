"""Fit a candidate relevance model and test one admission on aligned topic panels."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from audit_wide_vote_contributions import member
from eval_event_date_windows import write_gz
from eval_fifth_ranker import official_inputs, prepare
from replay_anchor_cap import z
from replay_conditional_rank_fusion import key, labels, measure, rank_feature
from replay_incumbent_evidence import effect, macro


ARMS = ('admit', 'low_coverage')
FIT = {'initial-186', 'confirm-155', 'reserve-109'}


def archived_queries(root, population):
    if population == 'official':
        return official_inputs(root)
    archive = root/'paired-membership-20261127.tar.gz'
    discovery = member(archive, 'paired-membership-641-final-20261127.json')['per_query']
    confirmation = member(archive, 'paired-membership-242-20261127.json')['per_query']
    if population == 'reserve':
        source = [r for r in discovery if r['panel'] == 'reserve-191']
    else:
        source = [r for r in discovery if r['panel'] in FIT] + confirmation
    return [{k:r[k] for k in ('domain', 'qid', 'panel')} |
            {'current': r['rankings']['current'], 'prior_score': r['scores']['current']}
            for r in source]


def build(args):
    rows = archived_queries(args.archive_dir, args.population)
    args.official = args.population == 'official'
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain']].append(row)
    split = 'dev' if args.official else 'train'
    for domain, group in groups.items():
        dense = {r['qid']: [] for r in group}
        with (args.runs/f'dense_mxbai_{domain}_{split}.trec').open(encoding='utf-8') as stream:
            for line in stream:
                qid, _, doc, rank, _, _ = line.split()
                if qid in dense and int(rank) <= 60:
                    dense[qid].append((int(rank), doc))
        for row in group:
            ranking = [d for _, d in sorted(dense[row['qid']])]
            if len(ranking) != 60 or len(set(ranking)) != 60:
                raise ValueError(f'Missing fixed pool for {key(row)}')
            row['docs'] = list(dict.fromkeys([*row['current'], *ranking]))
            row['sources'] = [ranking]
    return prepare(args, rows)


def features(row):
    docs = row['docs']
    raw = np.array(row['new_scores'])
    prior = rank_feature(docs, row['current'])
    dense = rank_feature(docs, row['sources'][0])
    standard = z(raw)
    member_mask = prior>0
    return np.column_stack((raw, standard, prior, dense, member_mask, standard*member_mask))


def fit(rows, gold):
    counts = Counter(r['domain'] for r in rows)
    inputs = np.concatenate([features(r) for r in rows])
    outputs = np.array([d in gold[key(r)] for r in rows for d in r['docs']])
    weights = np.array([1/(len(r['docs'])*counts[r['domain']]) for r in rows for _ in r['docs']])
    weights /= weights.mean()
    estimator = LogisticRegression(C=.3, max_iter=300)
    with threadpool_limits(2):
        estimator.fit(inputs, outputs, sample_weight=weights)
    return estimator


def rank(row, model, arms):
    current, docs = row['current'], row['docs']
    probabilities = dict(zip(docs, model.predict_proba(features(row))[:, 1], strict=True))
    outsider = max((d for d in docs if d not in current), key=probabilities.__getitem__)
    estimated_count = sum(probabilities[d] for d in current)
    approve = probabilities[outsider] > probabilities[current[9]]
    proposed = current[:9] + [outsider] if approve else current
    orders = {'current': current, 'admit': proposed,
              'low_coverage': proposed if estimated_count < 1 else current}
    return {k: row[k] for k in ('domain', 'qid', 'panel', 'prior_score')} | {
        'rankings': {name: orders[name] for name in ('current', *arms)}, 'estimated_count': estimated_count,
        'marginal_relevance': probabilities[outsider]-probabilities[current[9]]}


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def evaluate(args):
    screen = read(args.screen_scores)
    learned = [r for r in screen['per_query'] if r['panel'] in FIT]
    if len(learned) != 450:
        raise ValueError('Unexpected fit population')
    model = fit(learned, labels(learned, args.data_root))
    if args.selection:
        lock = read(args.selection)
        test = read(args.test_scores)
        previous_size = 191 if args.official else 242
        if (not lock['eligible'] or lock['queries'] != previous_size or lock['selected'] not in ARMS
                or screen['model'] != test['model'] or screen['model'] != lock['model']):
            raise ValueError('Invalid locked comparison')
        arms = (lock['selected'],)
        population = 'official-519' if args.official else 'reserve-191'
        targets = test['per_query']
    else:
        arms, population = ARMS, 'reserve-242'
        targets = [r for r in screen['per_query'] if r['panel'] not in FIT]
    expected = {'reserve-242': 242, 'reserve-191': 191, 'official-519': 519}[population]
    if len(targets) != expected or len({key(r) for r in targets}) != expected or {r['panel'] for r in targets} != {population}:
        raise ValueError('Unexpected evaluation IDs')
    if args.selection and {key(r) for r in targets} & {key(r) for r in lock['per_query']}:
        raise ValueError('Training-side screening and final evaluation topics overlap')
    rows = [rank(row, model, arms) for row in targets]
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    rows = measure(rows, labels(rows, args.data_root, 'dev' if args.official else 'train'))
    scores = {name: macro(rows, name) for name in ('current', *arms)}
    contrasts = {name: effect(rows, name, 'current') for name in arms}
    passing = [a for a in arms if min(contrasts[a]['point'], contrasts[a]['minimum_leave_one_domain_out']) > 0]
    if not args.selection and 'low_coverage' in passing and scores['low_coverage'] <= scores['admit']:
        passing.remove('low_coverage')
    winner = max(passing, key=scores.get, default=None)
    low = [r for r in rows if r['estimated_count'] < 1]
    routing = {'estimated_below_one': len(low), 'zero_reference_queries': sum(r['scores']['current'] == 0 for r in rows),
               'zero_reference_in_low_group': sum(r['scores']['current'] == 0 for r in low),
               'changed_queries': {a: sum(r['rankings'][a] != r['rankings']['current'] for r in rows) for a in arms}}
    return {'queries': expected, 'scores': scores, 'contrasts': contrasts, 'selected': winner,
            'eligible': winner is not None, 'rank_fingerprint': fingerprint, 'model': screen['model'],
            'routing': routing, 'per_query': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    prep = commands.add_parser('prepare')
    for arg in ('archive-dir', 'runs'):
        prep.add_argument(f'--{arg}', type=Path, required=True)
    prep.add_argument('--population', choices=('screen', 'reserve', 'official'), default='screen')
    test = commands.add_parser('evaluate')
    test.add_argument('--screen-scores', type=Path, required=True)
    test.add_argument('--selection', type=Path)
    test.add_argument('--test-scores', type=Path)
    test.add_argument('--official', action='store_true')
    for command in (prep, test):
        for arg in ('data-root', 'output'):
            command.add_argument(f'--{arg}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    if args.stage == 'evaluate' and (bool(args.selection) != bool(args.test_scores) or args.official and not args.selection):
        parser.error('A locked test requires both prior selection and test scores.')
    result = build(args) if args.stage == 'prepare' else evaluate(args)
    write_gz(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'per_query'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
