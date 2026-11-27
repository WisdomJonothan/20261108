"""Replay a fixed independent cross-encoder signal and two agreement controls."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from eval_event_date_windows import write_gz
from replay_anchor_cap import z
from replay_conditional_rank_fusion import key, labels, measure, rank_feature
from replay_incumbent_evidence import effect, macro


ARMS = ('large', 'small', 'mean', 'agreement')


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def order(values, current):
    return [current[0], *[current[i] for i in sorted(range(1, 10), key=lambda i: -values[i])]]


def shared_agreement(values_a, values_b):
    indices = []
    for i in range(1, 10):
        target = len(indices)
        while target and values_a[i] > values_a[indices[target-1]] and values_b[i] > values_b[indices[target-1]]:
            target -= 1
        indices.insert(target, i)
    return [0, *indices]


def ranks(previous, new, arms):
    if previous['current'] != new['current'] or previous['prior_score'] != new['prior_score']:
        raise ValueError(f'The maintained ranking changed for {key(new)}')
    current = new['current']
    prior = 3*z(rank_feature(current, current))
    raw_a = dict(zip(previous['docs'], previous['new_scores'], strict=True))
    raw_b = dict(zip(new['docs'], new['new_scores'], strict=True))
    a, b = [z([raw[d] for d in current]) + prior for raw in (raw_a, raw_b)]
    proposed = {'current': current, 'large': order(a, current), 'small': order(b, current),
                'mean': order((a+b)/2, current), 'agreement': [current[i] for i in shared_agreement(a, b)]}
    return {k: new[k] for k in ('domain', 'qid', 'panel', 'prior_score')} | {
        'rankings': {k: proposed[k] for k in ('current', *arms)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--large', type=Path, nargs='+', required=True)
    for k in ('small', 'output', 'data-root'):
        parser.add_argument(f'--{k}', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    large_inputs = [read(p) for p in args.large]
    small_input = read(args.small)
    metadata = {'large_model': large_inputs[0]['model'], 'small_model': small_input['model']}
    lock = read(args.selection) if args.selection else None
    if lock and (not lock['eligible'] or any(lock[k] != v for k,v in metadata.items())):
        raise ValueError('Invalid confirmation lock')
    selected = lock['selected'] if lock else None
    arms = tuple(dict.fromkeys(('large', 'small', selected))) if lock else ARMS
    old = {key(r): r for src in large_inputs for r in src['per_query']}
    new = {key(r): r for r in small_input['per_query']}
    if old.keys() != new.keys() or len(old) != (191 if lock else 692):
        raise ValueError('Expected aligned and complete populations')
    if lock and set(old) & {key(r) for r in lock['evaluated_rankings']}:
        raise ValueError('The discovery and confirmation topics overlap')
    candidate_rows = [ranks(old[k], row, arms) for k, row in new.items()]
    fingerprint = hashlib.sha256(json.dumps(candidate_rows, sort_keys=True).encode()).hexdigest()
    rows = measure(candidate_rows, labels(candidate_rows, args.data_root))
    scores = {name: macro(rows, name) for name in ('current', *arms)}
    contrasts = {arm: effect(rows, arm, 'current') for arm in arms}
    if lock:
        delta = contrasts[selected]
        matched = ['large'] if selected == 'small' else ['large', 'small']
        eligible = min(delta['point'], delta['minimum_leave_one_domain_out'],
                       *(scores[selected]-scores[a] for a in matched)) > 0
    else:
        panels = sorted({r['panel'] for r in rows})
        viable = []
        for arm in ('small', 'mean', 'agreement'):
            delta = contrasts[arm]
            delta['panel_omissions'] = {p: macro(subset, arm)-macro(subset, 'current')
                                        for p in panels if (subset := [r for r in rows if r['panel'] != p])}
            if min(delta['point'], delta['minimum_leave_one_domain_out'], *delta['panel_omissions'].values()) <= 0:
                continue
            peers = ['large'] if arm == 'small' else ['large', 'small']
            if all(scores[arm] > scores[p] for p in peers):
                viable.append(arm)
        selected = max(viable, key=scores.get, default=None)
        eligible = selected is not None
    report = {'eligible': eligible, 'selected': selected, 'queries': len(rows),
              'scores': scores, 'contrasts': contrasts, 'rank_fingerprint': fingerprint, **metadata}
    write_gz(args.output, report | {'evaluated_rankings': rows, 'small_scores': small_input['per_query']})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
