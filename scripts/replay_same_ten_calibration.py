"""Calibrate a fixed independent-score adjustment with disjoint evaluation panels."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from eval_event_date_windows import write_gz
from replay_anchor_cap import z
from replay_conditional_rank_fusion import key, labels, measure, rank_feature
from replay_incumbent_evidence import effect, macro


GRID = tuple(f'{shape}_{weight:g}' for shape in ('raw', 'sigmoid', 'rank')
             for weight in (.125, .25, .375, .5))
CONTROL = 'raw_0.25'


def read(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def ranks(row, arms):
    current = row['current']
    raw = dict(zip(row['docs'], row['new_scores'], strict=True))
    logits = np.array([raw[d] for d in current])
    prior = z(rank_feature(current, current))
    ordered = sorted(current, key=lambda doc: -raw[doc])
    signals = {'raw': z(logits), 'sigmoid': z(expit(logits)),
               'rank': z(rank_feature(current, ordered))}
    output = {'current': current}
    for arm in arms:
        shape, text = arm.split('_')
        weight = float(text)
        blend = (1-weight)*prior + weight*signals[shape]
        remainder = sorted(range(1, 10), key=lambda i: -blend[i])
        output[arm] = [current[0], *[current[i] for i in remainder]]
    return {k: row[k] for k in ('domain', 'qid', 'panel', 'prior_score')} | {'rankings': output}


def choose(rows):
    scores = {arm: macro(rows, arm) for arm in ('current', *GRID)}
    return max(scores, key=scores.get)


def train(rows):
    panels = sorted({r['panel'] for r in rows})
    scores = {a: macro(rows, a) for a in ('current', *GRID)}
    viable = []
    for arm in GRID:
        delta = effect(rows, arm, 'current')
        if min(delta['point'], delta['minimum_leave_one_domain_out']) <= 0:
            continue
        for panel in panels:
            rest = [r for r in rows if r['panel'] != panel]
            if macro(rest, arm) <= macro(rest, 'current'):
                break
        else:
            viable.append(arm)
    selected = max(viable, key=lambda name: scores[name], default=None)
    choices = {p: choose([r for r in rows if r['panel'] != p]) for p in panels}
    for row in rows:
        row['scores']['cross_panel'] = row['scores'][choices[row['panel']]]
        row['rankings']['cross_panel'] = row['rankings'][choices[row['panel']]]
    cross = effect(rows, 'cross_panel', 'current')
    contrasts = {'cross_panel': cross}
    if selected is not None:
        contrasts[selected] = effect(rows, selected, 'current')
    eligible = (selected is not None and scores[selected] > scores[CONTROL] and cross['point'] > 0)
    return {'selected': selected, 'eligible': eligible, 'scores': scores | {'cross_panel': macro(rows, 'cross_panel')},
            'fold_choices': choices, 'contrasts': contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', nargs='+', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    scores = [read(path) for path in args.scores]
    models = {s['model'] for s in scores}
    if len(models) != 1:
        raise ValueError('Expected a common checkpoint')
    model = models.pop()
    previous = read(args.lock) if args.lock else None
    if previous and (not previous['eligible'] or previous['model'] != model):
        raise ValueError('Expected a passing discovery lock with the same model')
    selected = previous['selected'] if previous else None
    arms = tuple(dict.fromkeys((CONTROL, selected))) if selected else GRID
    rows = [ranks(row, arms) for score in scores for row in score['per_query']]
    expected = {'reserve-191'} if previous else {'initial-186', 'confirm-155', 'reserve-109', 'reserve-242'}
    if {r['panel'] for r in rows} != expected or len({key(r) for r in rows}) != (191 if previous else 692):
        raise ValueError('Unexpected calibration or transfer population')
    if previous and {key(r) for r in rows} & {key(r) for r in previous['per_query']}:
        raise ValueError('Training and confirmation populations overlap')
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    measured = measure(rows, labels(rows, args.data_root))
    if previous:
        delta = effect(measured, selected, 'current')
        matched = effect(measured, selected, CONTROL)
        result = {'selected': selected,
                  'eligible': min(delta['point'], delta['minimum_leave_one_domain_out'], matched['point']) > 0,
                  'scores': {a: macro(measured, a) for a in ('current', *arms)},
                  'contrasts': {'current': delta, CONTROL: matched}}
    else:
        result = train(measured)
    result |= {'model': model, 'queries': len(rows), 'rank_fingerprint': fingerprint}
    write_gz(args.output, result | {'per_query': measured})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
