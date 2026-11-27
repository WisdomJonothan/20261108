"""Compare calibrated protected suffix rankings across fixed training panels."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from eval_coverage_routed_admission import FIT, features, fit, read
from eval_event_date_windows import write_gz
from replay_conditional_rank_fusion import key, labels, measure
from replay_incumbent_evidence import effect, macro


ARMS = ('linear_inside', 'linear_union', 'trees_inside', 'trees_union')


def build_model(data, gold, method):
    if method == 'linear':
        return fit(data, gold)
    counts = Counter(r['domain'] for r in data)
    x = np.concatenate([features(r) for r in data])
    y = [d in gold[key(r)] for r in data for d in r['docs']]
    weights = np.array([1/(len(r['docs'])*counts[r['domain']]) for r in data for _ in r['docs']])
    weights /= weights.mean()
    model = HistGradientBoostingClassifier(max_iter=110, max_leaf_nodes=7, min_samples_leaf=50,
                                          l2_regularization=7, early_stopping=False, random_state=431)
    with threadpool_limits(2):
        return model.fit(x, y, sample_weight=weights)


def predict(data, models, arms):
    output = []
    with threadpool_limits(2):
        for row in data:
            current, docs = row['current'], row['docs']
            score = {name: dict(zip(docs, model.predict_proba(features(row))[:, 1], strict=True))
                     for name, model in models.items()}
            orders = {'current': current}
            for arm in arms:
                name, scope = arm.split('_')
                head = current[:3]
                pool = current if scope == 'inside' else docs
                suffix = sorted((d for d in pool if d not in head), key=lambda d: -score[name][d])
                orders[arm] = head + suffix[:7]
            output.append({k: row[k] for k in ('domain', 'qid', 'panel', 'prior_score')} | {'rankings': orders})
    return output


def run(args):
    source = read(args.scores)
    data = source['per_query']
    training = [r for r in data if r['panel'] in FIT]
    if len(training) != 450 or len({key(r) for r in data}) != 692:
        raise ValueError('Expected the exact aligned data and disjoint panels')
    gold = labels(training, args.data_root)
    if args.selection:
        prior = read(args.selection)
        arm = prior['selected']
        if prior['model'] != source['model'] or arm not in ARMS or not prior['eligible']:
            raise ValueError('Expected a passing selection from discovery')
        targets = [r for r in data if r['panel'] not in FIT]
        if len(targets) != 242 or {r['panel'] for r in targets} != {'reserve-242'}:
            raise ValueError('Expected the registered 242')
        method = arm.split('_')[0]
        rows = predict(targets, {method: build_model(training, gold, method)}, (arm,))
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        rows = measure(rows, labels(targets, args.data_root))
        contrast = effect(rows, arm, 'current')
        eligible = min(contrast['point'], contrast['minimum_leave_one_domain_out']) > 0
        return {'model': source['model'], 'selected': arm, 'eligible': eligible,
                'queries': len(rows), 'scores': {a: macro(rows, a) for a in ('current', arm)},
                'contrasts': {arm: contrast}, 'rank_fingerprint': digest, 'per_query': rows}
    frozen = []
    for panel in sorted(FIT):
        fit_rows = [r for r in training if r['panel'] != panel]
        targets = [r for r in training if r['panel'] == panel]
        models = {method: build_model(fit_rows, gold, method) for method in ('linear', 'trees')}
        frozen.extend(predict(targets, models, ARMS))
    rows = measure(frozen, gold)
    scores = {a: macro(rows, a) for a in ('current', *ARMS)}
    stats = {a: effect(rows, a, 'current') for a in ARMS}
    eligible = []
    for a in ARMS:
        stats[a]['panel_omissions'] = {p: macro(part, a)-macro(part, 'current') for p in sorted(FIT)
                                       if (part := [r for r in rows if r['panel'] != p])}
        if min(stats[a]['point'], stats[a]['minimum_leave_one_domain_out'],
               *stats[a]['panel_omissions'].values()) > 0:
            eligible.append(a)
    return {'model': source['model'], 'selected': max(eligible, key=scores.get, default=None),
            'eligible': bool(eligible), 'queries': len(rows), 'scores': scores, 'contrasts': stats,
            'per_query': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scores', 'data-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    result = run(args)
    write_gz(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'per_query'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
