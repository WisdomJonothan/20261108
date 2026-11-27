"""Test probabilistic rank fusion by holding out complete training panels."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from audit_wide_vote_contributions import member
from official.scorer import load_qrels, ndcg_at_k
from replay_anchor_cap import z
from replay_incumbent_evidence import effect, macro


SOURCES = ('current', 'reasoned', 'verified', 'pointwise')
PANELS = {
    'initial-186': ('evidence_ranking_consensus_20261126', 'evidence-consensus-extension-holdout.json'),
    'confirm-155': ('temporal_conflict_disjoint_confirmation_20261126', 'confirmation-consensus.json'),
    'reserve-109': ('near-adapter-reserve-20261126', 'consensus.json'),
    'reserve-242': ('locked-local-vote-reserve-242-20261126', 'consensus.json'),
}
ARMS = ('linear_inside', 'linear_union', 'trees_inside', 'trees_union')


def key(row):
    return row['domain'], row['qid']


def read_rows(archive_dir, panels):
    previous = archive_dir / 'paired-membership-20261127.tar.gz'
    refs = {}
    for name in ('641-final', '242'):
        refs.update({key(row): row for row in member(previous, f'paired-membership-{name}-20261127.json')['per_query']
                     if row['panel'] in panels})
    result = []
    for panel in panels:
        archive, name = PANELS[panel]
        rows = member(archive_dir / f'{archive}.tar.gz', name)['per_query']
        for row in rows:
            ref = refs[key(row)]
            if row['split'] != 'train' or ref['panel'] != panel:
                raise ValueError(f'Unexpected population for {key(row)}')
            current = ref['rankings']['current']
            sources = [row['rankings'][name] for name in SOURCES]
            docs = list(dict.fromkeys([*current, *(doc for order in sources for doc in order)]))
            result.append({'domain': row['domain'], 'qid': row['qid'], 'panel': panel,
                           'current': current, 'prior_score': ref['scores']['current'],
                           'docs': docs, 'sources': sources,
                           'features': make_features(docs, current, sources)})
    keys = {key(row) for row in result}
    if keys != refs.keys() or len(result) != len(keys):
        raise ValueError('Rank sources differ from the fixed reference population')
    return result


def rank_feature(docs, ranking):
    lookup = {doc: 1 / np.log2(i + 2) for i, doc in enumerate(ranking)}
    return np.array([lookup.get(doc, 0.0) for doc in docs])


def make_features(docs, current, sources):
    features = np.column_stack([rank_feature(docs, order) for order in sources])
    present = (features > 0).sum(axis=1)
    top3 = (features >= .5).sum(axis=1)
    base = rank_feature(docs, current)
    overlap = [len(set(sources[i]) & set(sources[j])) / 10
               for i, j in ((0, 3), (1, 2), (2, 3))]
    return np.column_stack((base, features, present, top3, features.mean(axis=1),
                            features.max(axis=1), np.tile(overlap, (len(docs), 1))))


def labels(rows, root, split='train'):
    cache = {d: load_qrels(root / d / f'qrels_{split}.txt') for d in {r['domain'] for r in rows}}
    return {key(r): cache[r['domain']][r['qid']] for r in rows}


def fit(rows, gold, names):
    sizes = Counter(r['domain'] for r in rows)
    x = np.concatenate([r['features'] for r in rows])
    y = np.array([d in gold[key(r)] for r in rows for d in r['docs']])
    w = np.array([1 / (len(r['docs']) * sizes[r['domain']]) for r in rows for _ in r['docs']])
    w /= w.mean()
    estimators = {
        'linear': LogisticRegression(C=.2, max_iter=300),
        'trees': HistGradientBoostingClassifier(max_iter=90, max_leaf_nodes=7, min_samples_leaf=30,
                                                l2_regularization=5, early_stopping=False, random_state=17),
    }
    with threadpool_limits(2):
        return {name: estimators[name].fit(x, y, sample_weight=w) for name in names}


def predict(rows, estimators, arms):
    output = []
    with threadpool_limits(2):
        for row in rows:
            docs, current = row['docs'], row['current']
            rankings = {'current': current}
            values = {name: model.decision_function(row['features']) for name, model in estimators.items()}
            for arm in arms:
                name, scope = arm.split('_')
                pool = current if scope == 'inside' else docs
                indices = [docs.index(d) for d in pool]
                total = 3 * z(rank_feature(pool, current)) + z(values[name][indices])
                order = sorted(range(len(pool)), key=lambda i: -total[i])
                if scope == 'union':
                    ranking = [current[0], *[pool[i] for i in order if pool[i] != current[0]][:9]]
                else:
                    ranking = [pool[i] for i in order[:10]]
                rankings[arm] = ranking
            output.append({k: row[k] for k in ('domain', 'qid', 'panel', 'prior_score')} |
                          {'rankings': rankings})
    return output


def measure(rows, gold):
    for row in rows:
        row['scores'] = {arm: ndcg_at_k(order, gold[key(row)]) for arm, order in row['rankings'].items()}
        if abs(row['scores']['current'] - row.pop('prior_score')) > 1e-12:
            raise ValueError(f'Official scorer did not reproduce the baseline for {key(row)}')
    return rows


def run_discovery(args):
    panels = tuple(p for p in PANELS if p != 'reserve-242')
    records = read_rows(args.archive_dir, panels)
    gold = labels(records, args.data_root)
    ranks = []
    for panel in panels:
        fit_rows = [r for r in records if r['panel'] != panel]
        held = [r for r in records if r['panel'] == panel]
        ranks.extend(predict(held, fit(fit_rows, gold, ('linear', 'trees')), ARMS))
    rows = measure(ranks, gold)
    contrasts = {a: effect(rows, a, 'current') for a in ARMS}
    for arm, stats in contrasts.items():
        stats['panel_omissions'] = {p: macro(skip, arm) - macro(skip, 'current')
                                    for p in panels if (skip := [r for r in rows if r['panel'] != p])}
    eligible = [a for a in ARMS if min(contrasts[a]['point'], contrasts[a]['minimum_leave_one_domain_out'],
                                      *contrasts[a]['panel_omissions'].values()) > 0]
    selected = max(eligible, key=lambda arm: contrasts[arm]['point'], default=None)
    return {'selected': selected, 'queries': len(rows),
            'scores': {a: macro(rows, a) for a in ('current', *ARMS)},
            'contrasts': contrasts, 'per_query': rows}


def run_confirmation(args):
    with gzip.open(args.selection, 'rt', encoding='utf-8') as stream:
        selection = json.load(stream)['selected']
    if selection not in ARMS:
        raise ValueError('No eligible selection to confirm')
    development = read_rows(args.archive_dir, tuple(p for p in PANELS if p != 'reserve-242'))
    holdout = read_rows(args.archive_dir, ('reserve-242',))
    if {key(r) for r in development} & {key(r) for r in holdout}:
        raise ValueError('Overlapping discovery and confirmation IDs')
    models = fit(development, labels(development, args.data_root), (selection.split('_')[0],))
    ranks = predict(holdout, models, (selection,))
    fingerprint = hashlib.sha256(json.dumps(ranks, sort_keys=True).encode()).hexdigest()
    rows = measure(ranks, labels(holdout, args.data_root))
    change = effect(rows, selection, 'current')
    return {'selected': selection, 'queries': len(rows), 'rank_fingerprint': fingerprint,
            'eligible': min(change['point'], change['minimum_leave_one_domain_out']) > 0,
            'scores': {a: macro(rows, a) for a in ('current', selection)},
            'contrasts': {selection: change}, 'per_query': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('archive-dir', 'data-root', 'output'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    result = run_confirmation(args) if args.selection else run_discovery(args)
    with gzip.open(args.output, 'xt', encoding='utf-8') as stream:
        json.dump(result, stream, allow_nan=False, separators=(',', ':'))
    print(json.dumps({k:v for k,v in result.items() if k != 'per_query'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
