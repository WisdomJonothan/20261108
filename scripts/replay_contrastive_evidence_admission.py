"""Test whether matched textual corroboration realizes contrastive candidate gains."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from audit_wide_vote_contributions import member
from coverage_cf_bound import read_steps
from eval_event_date_windows import write_gz
from eval_fifth_ranker import prepare
from replay_conditional_rank_fusion import key, labels, measure
from replay_incumbent_evidence import effect, macro


ARMS = ('original', 'original_gap_mix', 'two_sided')


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def build(args):
    source = read(args.candidates)
    archived = {key(r): r for r in member(args.archive, 'wide-vote-reserve-result.json')['per_query']}
    if not source['eligible'] or set(source['summary']) != {'reserve'}:
        raise ValueError('The two-sided candidate study must pass before reranking')
    rows = []
    for row in source['per_query']:
        old = archived[key(row)]
        current = row['current']
        if current != old['rankings']['current']:
            raise ValueError(f'Frozen rankings differ on {key(row)}')
        lists = [row['offers'][arm] for arm in ARMS]
        docs = list(dict.fromkeys([current[-1], *(d for items in lists for d in items)]))
        if any(len(offers) != 4 or set(offers) & set(current) for offers in lists):
            raise ValueError('Expected four distinct outsiders for each arm')
        rows.append({k: row[k] for k in ('domain', 'qid', 'panel', 'current')} |
                    {'docs': docs, 'sources': lists, 'prior_score': old['scores']['current']})
    args.official = False
    return prepare(args, rows)


def comparisons(args):
    large, small = (read(path) for path in (args.large, args.small))
    large_rows, small_rows = ({key(r): r for r in view['per_query']} for view in (large, small))
    if len(large_rows) != 191 or large_rows.keys() != small_rows.keys():
        raise ValueError('Expected aligned scores for every reserve topic')
    rows = []
    for k, row in large_rows.items():
        other = small_rows[k]
        if row['current'] != other['current'] or row['docs'] != other['docs'] or row['sources'] != other['sources']:
            raise ValueError(f'Unexpected candidate inputs on {k}')
        scores = [dict(zip(r['docs'], r['new_scores'], strict=True)) for r in (row, other)]
        current = row['current']
        order = {'current': current}
        for arm, offers in zip(ARMS, row['sources'], strict=True):
            first, second = (max(offers, key=score.__getitem__) for score in scores)
            if first == second and all(score[first] > score[current[-1]] for score in scores):
                order[arm] = current[:9] + [first]
            else:
                order[arm] = current
        rows.append({k:row[k] for k in ('domain','qid','prior_score')} | {'rankings': order})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    rows = measure(rows, labels(rows, args.data_root))
    steps = {d: read_steps(args.data_root/d/'qrels_steps_train.txt') for d in {r['domain'] for r in rows}}
    totals = {a: {'changed': 0, 'gained_steps': 0, 'lost_steps': 0} for a in ARMS}
    for row in rows:
        step_sets = steps[row['domain']].get(row['qid'], {})
        coverage = {name: {i for i, docs in step_sets.items() if not docs.isdisjoint(order)}
                    for name, order in row['rankings'].items()}
        for arm in ARMS:
            totals[arm]['changed'] += row['rankings'][arm] != row['rankings']['current']
            totals[arm]['gained_steps'] += len(coverage[arm]-coverage['current'])
            totals[arm]['lost_steps'] += len(coverage['current']-coverage[arm])
    estimates = {a: macro(rows, a) for a in ('current', *ARMS)}
    contrasts = {a: effect(rows, 'two_sided', a) for a in ('current', 'original', 'original_gap_mix')}
    result = {'queries': len(rows), 'models': [large['model'], small['model']], 'scores': estimates,
              'contrasts': contrasts, 'edits': totals, 'fingerprint': digest, 'per_query': rows,
              'raw_scores': {'large': large['per_query'], 'small': small['per_query']}}
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_query','raw_scores')}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    prep = stages.add_parser('prepare')
    prep.add_argument('--candidates', type=Path, required=True)
    prep.add_argument('--archive', type=Path, required=True)
    score = stages.add_parser('score')
    score.add_argument('--large', type=Path, required=True)
    score.add_argument('--small', type=Path, required=True)
    for stage in (prep, score):
        for arg in ('data-root','output'):
            stage.add_argument(f'--{arg}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    write_gz(args.output, build(args) if args.stage == 'prepare' else comparisons(args))


if __name__ == '__main__':
    main()
