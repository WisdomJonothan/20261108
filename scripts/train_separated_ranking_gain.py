"""Calibrate gains from retained rank movement and top-ten membership separately."""
import argparse
from collections import Counter
from glob import glob
import json
from math import log2
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import Ridge

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges
from official.scorer import load_qrels, ndcg_at_k
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import discounted_changes, index
from replay_passage_admission import observations
from replay_wide_pointwise_vote import replay


ACTIONS = ('current', 'reorder_current_membership', 'local_candidate_scale')
DISCOUNT = {rank: 1. / log2(rank + 2) for rank in range(10)}
FEATURES = ('count', 'upstream_rank', 'original_pointwise_rank', 'verified_evidence',
            'small_model_percentile', 'frozen_8b_percentile')
PENALTY = 32.


def paths(patterns):
    files = [Path(path) for pattern in patterns for path in sorted(glob(pattern))]
    if not files or len(set(files)) != len(files):
        raise ValueError(f'Expected unique inputs matching {patterns}')
    return files


def percentile(values):
    order = sorted(values, key=values.__getitem__, reverse=True)
    middle = (len(order) - 1) / 2
    return {doc: (middle - i) / max(middle, 1.) for i, doc in enumerate(order)}


def support(scales, original, scope):
    first = {doc: 1. / (1 + rank) for rank, doc in enumerate(original['current'])}
    point = {doc: 1. / (1 + rank) for rank, doc in enumerate(original['pointwise'])}
    values, scores, grades = scales
    return {doc: np.array([1., first.get(doc, 0.), point.get(doc, 0.),
                           grades.get(doc, 0.), values[doc], scores[doc]]) for doc in scope}


def delta_features(before, after, support_by_doc):
    left = {doc: DISCOUNT[i] for i, doc in enumerate(before)}
    right = {doc: DISCOUNT[i] for i, doc in enumerate(after)}
    shared = left.keys() & right.keys()
    move = np.zeros(len(FEATURES))
    admission = np.zeros(len(FEATURES))
    for doc in dict.fromkeys([*before, *after]):
        weight = right.get(doc, 0.) - left.get(doc, 0.)
        target = move if doc in shared else admission
        target += weight * support_by_doc[doc]
    return np.array([move, admission])


def materialize(spec):
    args = SimpleNamespace(
        **{name: Path(spec[name]) for name in ('manifest', 'consensus', 'reference')},
        **{name: paths(spec[name]) for name in ('pointwise', 'evidence', 'scan')},
        retain_position_prior=True, protect_membership=True, stable_across_scales=False,
        taper_vote=False, guard_admission=False,
    )
    votes = index(load(args.consensus))
    proof = observations(args.evidence)
    small = candidate_judges(args.scan)[0]['pretrained']
    large = candidate_judges(args.pointwise)[0]['pretrained']
    topics = []
    for row in replay(args):
        key = row['domain'], row['qid']
        ranks = {action: row['rankings'][action] for action in ACTIONS}
        docs = dict.fromkeys(doc for arm in ACTIONS for doc in ranks[arm])
        signals = (
            percentile(small[key]['scores']),
            percentile(large[key]['scores']),
            {doc: obs['reasoned'] / 3. for doc, obs in proof[key].items() if obs['verified']},
        )
        components = support(signals, votes[key]['rankings'], docs)
        features = [delta_features(ranks['current'], ranks[arm], components) for arm in ACTIONS[1:]]
        topics.append({field: row[field] for field in ('domain', 'qid', 'split')}
                      | {'panel': spec['name'], 'rankings': ranks, 'features': np.array(features)})
    return topics


def judge(rows, data_root):
    labels = {}
    for row in rows:
        domain, qid = row['domain'], row['qid']
        if domain not in labels:
            labels[domain] = load_qrels(data_root / domain / 'qrels_train.txt')
        gold = labels[domain][qid]
        ranks = row['rankings']
        before = ranks['current']
        row['scores'] = {arm: ndcg_at_k(order, gold, 10) for arm, order in ranks.items()}
        row['target'] = []
        for arm in ACTIONS[1:]:
            effects = discounted_changes(before, ranks[arm], gold)
            row['target'].append((effects['retained_documents'],
                                  effects['added_documents'] + effects['removed_documents']))
    return rows


def fit(rows):
    counts = Counter(row['domain'] for row in rows)
    features = np.concatenate([row['features'] for row in rows])
    targets = np.concatenate([row['target'] for row in rows])
    weights = np.array([1. / counts[row['domain']] for row in rows for _ in ACTIONS[1:]])
    coefs = [Ridge(alpha=PENALTY, fit_intercept=False).fit(
        features[:, j], targets[:, j], sample_weight=weights/weights.mean()).coef_ for j in range(2)]
    return np.array(coefs)


def choose(row, coefs):
    gains = (row['features'] * coefs).sum(axis=(1, 2))
    return ACTIONS[int(np.argmax([0., *gains]))]


def crossfit(rows, field):
    predictions = {}
    for held in sorted({row[field] for row in rows}):
        model = fit([row for row in rows if row[field] != held])
        predictions.update({(row['domain'], row['qid']): choose(row, model)
                            for row in rows if row[field] == held})
    return predictions


def summarize(rows, predictions):
    paired = [(row['domain'], row['scores'][predictions[row['domain'], row['qid']]] - row['scores']['current'])
              for row in rows]
    effect = paired_delta(paired)
    groups = effect['per_domain']
    score = mean(mean(r['scores']['current'] for r in rows if r['domain'] == d) for d in groups)
    worst = min(mean(value for d, value in groups.items() if d != hold) for hold in groups)
    return {'baseline': score, 'score': score+effect['point'], 'change': effect,
            'minimum_leave_one_domain_out': worst, 'actions': Counter(predictions.values())}


def valid(result):
    return result['change']['point'] > 0 and result['minimum_leave_one_domain_out'] > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'data-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--locked', type=Path)
    parser.add_argument('--reserve-ids', type=Path)
    args = parser.parse_args()
    if args.output.exists() or bool(args.locked) != bool(args.reserve_ids):
        parser.error('Use a fresh output and provide both frozen fit and IDs for transfer')
    rows = [row for spec in load(args.sources)['panels'] for row in materialize(spec)]
    keys = {(row['domain'], row['qid']) for row in rows}
    if len(keys) != len(rows) or {r['split'] for r in rows} != {'train'}:
        raise ValueError('Expected distinct official training topics')
    if args.locked:
        lock = load(args.locked)
        if not lock['eligible'] or keys != {tuple(key) for key in load(args.reserve_ids)['queries']}:
            raise ValueError('Expected the eligible fixed fit on exactly the reserved topics')
        if keys.intersection(map(tuple, lock['fit_ids'])):
            raise ValueError('Reserved topics overlap with fitting')
        picks = {(r['domain'], r['qid']): choose(r, np.asarray(lock['coefficients'])) for r in rows}
        for row in rows:
            chosen = picks[row['domain'], row['qid']]
            ranks = row['rankings']
            row['rankings'] = {a: ranks[a] for a in ('current', chosen)}
        # Rank choices are frozen before their relevance judgments are loaded.
        qrels = {d: load_qrels(args.data_root / d / 'qrels_train.txt') for d, _ in keys}
        for row in rows:
            gold = qrels[row['domain']][row['qid']]
            row['scores'] = {a: ndcg_at_k(r, gold, 10) for a, r in row['rankings'].items()}
        result = {'transfer': summarize(rows, picks)}
        for row in rows:
            row['chosen'] = picks[row['domain'], row['qid']]
    else:
        if len(keys) != 641 or len({r['panel'] for r in rows}) != 4:
            raise ValueError('Expected the four earlier disjoint training-side comparisons')
        judge(rows, args.data_root)
        policies = {field: crossfit(rows, field) for field in ('domain', 'panel')}
        validation = {field: summarize(rows, policy) for field, policy in policies.items()}
        fixed = {action: summarize(rows, dict.fromkeys(keys, action)) for action in ACTIONS[1:]}
        eligible = all(valid(report) and report['score'] > fixed[ACTIONS[-1]]['score']
                       for report in validation.values())
        result = {'validation': validation, 'fixed': fixed, 'eligible': eligible,
                  'features': FEATURES, 'penalty': PENALTY, 'coefficients': fit(rows).tolist(),
                  'fit_ids': sorted(keys)}
        for row in rows:
            key = row['domain'], row['qid']
            row['crossfit_choices'] = {field: picked[key] for field, picked in policies.items()}
            row.pop('target')
    for row in rows:
        row.pop('features')
    result['per_query'] = rows
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, separators=(',', ':'))
    print({k: v for k, v in result.items() if k in ('eligible', 'validation', 'transfer')}, flush=True)


if __name__ == '__main__':
    main()
