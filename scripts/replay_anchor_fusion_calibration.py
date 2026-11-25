"""Test fusion calibration with the same 8B scores and 34/10-document pools."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index

PRIOR = np.array([.7, .3, 0.])
WEIGHTS = (.1, .2, .3, .4, .5, .6)
ARM_NAMES = ('fixed', 'shared_weight', 'route_weights', 'weighted_rrf',
             'pairwise_uniform', 'pairwise_discount')


def build(archive, paths, frozen, data, split):
    budgeted = {(r['domain'], r['qid']): r for r in archive[f'{split}_per_query']}
    pools = index(load(frozen))
    records = []
    seen = set()
    label_split = 'train' if split == 'oof' else 'dev'
    labels = {domain: read_qrels(data / domain / f'qrels_{label_split}.txt')
              for domain in {d for d, _ in budgeted}}
    for cohort, path in enumerate(paths):
        for (d, q), raw in index(load(path)).items():
            key = d, q
            if key in seen:
                raise ValueError(f'Overlapping query IDs for {key}')
            seen.add(key)
            saved = budgeted[key]
            base = dict(pools[key]['frozen_scores'])
            full = list(base)
            anchor = raw['scores']
            if len(full) != 100 or base.keys() != anchor.keys():
                raise ValueError(f'Incomplete full pool for {key}')
            fallback = saved['active']
            start = raw['arms']['base']['ranking'] if fallback else saved['accepted_orders']['full_top10']
            keep = set(start)
            if fallback:
                keep.update([doc for doc in full if doc not in keep][:24])
            indices = [i for i, doc in enumerate(full) if doc in keep]
            docs = [full[i] for i in indices]
            old = [base[doc] for doc in full]
            frozen_z = z([old[i] for i in indices]) if fallback else z(old)[indices]
            anchor_z = z([anchor[doc] for doc in docs])
            dense_z = z([1 / (61 + i) for i in indices])
            features = np.column_stack([frozen_z, anchor_z, dense_z])
            row = {'domain': d, 'qid': q, 'cohort': cohort, 'route': 'fallback' if fallback else 'accepted',
                   'docs': docs, 'gold': labels[d][q], 'features': features}
            expected = saved['rankings']['dense_24'] if fallback else saved['accepted_orders']['full_top10']
            if len(docs) != (34 if fallback else 10) or rank(row, PRIOR) != expected:
                raise ValueError(f'Fixed fusion differs from the former result for {key}')
            records.append(row)
    if seen != budgeted.keys():
        raise ValueError('Missing query scores')
    return records


def rank(row, weights):
    values = row['features'] @ weights
    return [row['docs'][i] for i in np.argsort(-values, kind='stable')[:10]]


def score(row, ranking):
    return ndcg10(ranking, row['gold'])


def weight(alpha):
    return np.array([1 - alpha, alpha, 0.])


def select_alpha(rows):
    def objective(alpha):
        return macro((r['domain'], score(r, rank(r, weight(alpha)))) for r in rows)

    return max(WEIGHTS, key=lambda alpha: (objective(alpha), -abs(alpha - .3)))


def rrf(row):
    features = row['features']
    first = rankdata(-features[:, 0], method='average')
    second = rankdata(-features[:, 1], method='average')
    values = .7 / (60 + first) + .3 / (60 + second)
    return [row['docs'][i] for i in np.argsort(-values, kind='stable')[:10]]


def pairs(rows, discounted):
    diffs, weights, domains = [], [], []
    for row in rows:
        features = row['features']
        positive = np.array([doc in row['gold'] for doc in row['docs']])
        pos, neg = np.flatnonzero(positive), np.flatnonzero(~positive)
        if not len(pos) or not len(neg):
            continue
        differences = (features[pos, None] - features[None, neg]).reshape(-1, 3)
        if discounted:
            order = np.argsort(-(features @ PRIOR), kind='stable')
            discounts = np.zeros(len(features))
            discounts[order[:10]] = 1 / np.log2(np.arange(2, 12))
            w = np.abs(discounts[pos, None] - discounts[None, neg]).ravel()
        else:
            w = np.ones(len(differences))
        diffs.append(differences)
        weights.append(w / w.sum())
        domains.append(row['domain'])
    counts = Counter(domains)
    if not diffs:
        raise ValueError('No positive-negative pairs in the selected training queries')
    return np.concatenate(diffs), np.concatenate([
        w / (len(counts) * counts[d]) for w, d in zip(weights, domains)
    ])


def fit_pairwise(rows, discounted):
    x, weight = pairs(rows, discounted)
    strength = .10

    def loss(theta):
        difference = theta - PRIOR
        margin = x @ theta
        objective = weight @ np.logaddexp(0., -margin) + .5 * strength * (difference @ difference)
        gradient = x.T @ (-weight * expit(-margin)) + strength * difference
        return objective, gradient

    fitted = minimize(loss, PRIOR, jac=True, method='L-BFGS-B', bounds=[(0, None)] * 3)
    if not fitted.success:
        raise ValueError(f'Pairwise fit failed: {fitted.message}')
    return fitted.x.tolist()


def fit(rows):
    routes = {route: [r for r in rows if r['route'] == route] for route in ('accepted', 'fallback')}
    return {
        'shared_alpha': select_alpha(rows),
        'route_alpha': {name: select_alpha(group) for name, group in routes.items()},
        'pairwise_uniform': {name: fit_pairwise(group, False) for name, group in routes.items()},
        'pairwise_discount': {name: fit_pairwise(group, True) for name, group in routes.items()},
    }


def predict(rows, params):
    result = []
    for row in rows:
        route = row['route']
        orders = {
            'fixed': rank(row, PRIOR),
            'shared_weight': rank(row, weight(params['shared_alpha'])),
            'route_weights': rank(row, weight(params['route_alpha'][route])),
            'weighted_rrf': rrf(row),
            'pairwise_uniform': rank(row, params['pairwise_uniform'][route]),
            'pairwise_discount': rank(row, params['pairwise_discount'][route]),
        }
        scores = {name: score(row, docs) for name, docs in orders.items()}
        result.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'route')} |
                      {'rankings': orders, 'scores': scores})
    return result


def summarize(rows):
    summary = {}
    for arm in ARM_NAMES:
        gains = [(r['domain'], r['scores'][arm] - r['scores']['fixed']) for r in rows]
        summary[arm] = {
            'ndcg': macro((r['domain'], r['scores'][arm]) for r in rows),
            'change': paired_delta(gains),
            'contribution_by_route': {
                route: macro((r['domain'], delta if r['route'] == route else 0.)
                             for r, (_, delta) in zip(rows, gains))
                for route in ('accepted', 'fallback')
            },
        }
    return summary


def audit_routes(rows, arm):
    results = {}
    for route in ('accepted', 'fallback'):
        changes = [(r['domain'], r['scores'][arm] - r['scores']['fixed']
                    if r['route'] == route else 0.) for r in rows]
        paired = paired_delta(changes)
        values = paired['per_domain']
        leave_one_out = {held: mean(value for d, value in values.items() if d != held)
                         for held in values}
        results[route] = {
            'queries': sum(r['route'] == route for r in rows),
            'mix_one_route_ndcg': macro((r['domain'], r['scores'][arm] if r['route'] == route
                                         else r['scores']['fixed']) for r in rows),
            'change': paired,
            'leave_one_domain_out_range': [min(leave_one_out.values()), max(leave_one_out.values())],
            'positive_domain_count': sum(v > 1e-12 for v in values.values()),
            'negative_domain_count': sum(v < -1e-12 for v in values.values()),
        }
    return results


def weight_curve(rows):
    base = [score(row, rank(row, PRIOR)) for row in rows]
    curve = {}
    for alpha in WEIGHTS:
        candidate = [score(row, rank(row, weight(alpha))) for row in rows]
        curve[str(alpha)] = {
            route: paired_delta([
                (r['domain'], after - before if r['route'] == route else 0.)
                for r, before, after in zip(rows, base, candidate)
            ]) for route in ('accepted', 'fallback')
        }
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgeted', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--oof', type=Path, nargs=2, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--oof-frozen', type=Path, required=True)
    parser.add_argument('--dev-frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive = load(args.budgeted)
    train = build(archive, args.oof, args.oof_frozen, args.data_root, 'oof')
    dev = build(archive, [args.dev], args.dev_frozen, args.data_root, 'dev')
    crossfit, folds = [], []
    for cohort in (0, 1):
        params = fit([r for r in train if r['cohort'] != cohort])
        folds.append(params)
        crossfit.extend(predict([r for r in train if r['cohort'] == cohort], params))
    final = fit(train)
    official = predict(dev, final)
    result = {
        'crossfit_params_by_target_cohort': folds, 'pooled_train_params': final,
        'protocol': 'Freeze all candidate pools and judge scores. Select fusion weights on held-out '
                    'training cohorts or fit regularized pair losses on the same scores. '
                    'Inspect one-route changes as a retrospective diagnostic.',
        'oof': summarize(crossfit), 'dev': summarize(official),
        'route_audit': {
            split: {arm: audit_routes(rows, arm) for arm in ('pairwise_uniform', 'pairwise_discount')}
            for split, rows in (('oof', crossfit), ('dev', official))
        },
        'fixed_weight_curve': {'oof': weight_curve(train), 'dev': weight_curve(dev)},
        'oof_per_query': crossfit, 'dev_per_query': official,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, separators=(',', ':'))
        stream.write('\n')
    print('Final parameters:', final)
    for split in ('oof', 'dev'):
        for name, a in result[split].items():
            print(split, name, a['ndcg'], a['change']['ci95'], a['contribution_by_route'])


if __name__ == '__main__':
    main()
