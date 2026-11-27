"""Compare two fixed query truncation policies with the same top ten and scorer."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from eval_fifth_ranker import order_inside
from eval_event_date_windows import write_gz
from replay_conditional_rank_fusion import key, labels, measure
from replay_incumbent_evidence import effect, macro


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def evaluate(args):
    head, ends = read(args.head), read(args.ends)
    baseline = {key(r): r for r in head['per_query']}
    if head['model'] != ends['model'] or ends['truncation'] != 'first64-last128':
        raise ValueError('Expected a matched checkpoint and the fixed truncation policy')
    rows = []
    for row in ends['per_query']:
        first = baseline[key(row)]
        if first['current'] != row['current'] or first['docs'] != row['docs']:
            raise ValueError(f'Lists changed for {key(row)}')
        ranks = {'current': row['current']}
        for name, source in (('head_only', first), ('head_tail', row)):
            ranks[name] = order_inside(row['current'], dict(zip(source['docs'], source['new_scores'], strict=True)))
            if set(ranks[name]) != set(row['current']) or ranks[name][0] != row['current'][0]:
                raise ValueError('The correction must preserve membership and the first document')
        rows.append({k: row[k] for k in ('domain', 'qid', 'panel', 'prior_score')} | {'rankings': ranks})
    if len(rows) != len(baseline):
        raise ValueError('Population changed between the two scores')
    if args.discovery:
        discovery = read(args.discovery)
        if not discovery['eligible'] or discovery['model'] != head['model']:
            raise ValueError('Expected passing discovery with the same checkpoint')
        if len(rows) != 191 or {r['panel'] for r in rows} != {'reserve-191'}:
            raise ValueError('Expected the untouched 191-topic panel for this method')
        if {key(r) for r in rows} & {key(r) for r in discovery['per_query']}:
            raise ValueError('Discovery and confirmation populations overlap')
    elif len(rows) != 450:
        raise ValueError('Expected the fixed development population')
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    grade = measure(rows, labels(rows, args.data_root))
    comparisons = {name: effect(grade, 'head_tail', name) for name in ('current', 'head_only')}
    eligible = all(min(v['point'], v['minimum_leave_one_domain_out']) > 0 for v in comparisons.values())
    return {'model': head['model'], 'eligible': eligible, 'queries': len(grade), 'rank_fingerprint': digest,
            'rescored_queries': ends['rescored_queries'],
            'scores': {a: macro(grade, a) for a in ('current', 'head_only', 'head_tail')},
            'contrasts': comparisons, 'per_query': grade,
            'head_tail_raw': ends['per_query']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('head', 'ends', 'data-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--discovery', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    result = evaluate(args)
    write_gz(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_query', 'head_tail_raw')},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
