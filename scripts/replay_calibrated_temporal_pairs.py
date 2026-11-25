"""Calibrate which one-edit evidence proposal is likely to improve nDCG."""
import argparse
from collections import Counter
import json
from math import log2
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_temporal_pair_actions import POLICIES, choose as choose_direct
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


CALIBRATED = {f'calibrated_{l}_{g:g}': (l, g) for l in (40, 120) for g in (0., .005, .010)}
SETTINGS = (*POLICIES, *CALIBRATED)


def prepare(manifest, paths, data, split):
    raw = load(manifest)['queries']
    if {r['split'] for r in raw} != {split}:
        raise ValueError('Unexpected split')
    decisions = index({'per_query': [r for p in paths for r in load(p)['per_query']]})
    if decisions.keys() != {(r['domain'], r['qid']) for r in raw}:
        raise ValueError('The scores and proposal manifest differ')
    labels = {d: read_qrels(data / d / f'qrels_{split}.txt') for d, _ in decisions}
    result = []
    for row in raw:
        vote = decisions[row['domain'], row['qid']]['votes']
        gold = labels[row['domain']][row['qid']]
        base = ndcg10(row['reference'], gold)
        features, target = [], []
        for proposal, orders in zip(row['proposals'], vote, strict=True):
            slot = proposal['slot']
            impact = 1/log2(slot+2)
            head = proposal['type'] == 'head'
            if head:
                impact -= 1/log2(proposal['from_slot']+2)
            margin = [new-old for old, new, _ in orders]
            features.append([float(head), impact, np.mean(margin), min(margin),
                             abs(margin[0]-margin[1]), np.mean([v[2] for v in orders]),
                             *proposal['evidence_margin']])
            target.append(ndcg10(proposal['ranking'], gold)-base)
        result.append(row | {'features': features, 'gains': target, 'current': base, 'votes': vote})
    return result


def fit(rows, regularization):
    groups = Counter(r['domain'] for r in rows)
    weights = np.array([len(rows)/(len(groups)*groups[r['domain']]*len(r['proposals']))
                        for r in rows for _ in r['proposals']])
    x = np.asarray([x for r in rows for x in r['features']])
    target = np.asarray([y for r in rows for y in r['gains']])
    mean = np.average(x, axis=0, weights=weights)
    std = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights)).clip(.00001)
    y = np.average(target, weights=weights)
    xx = (x-mean)/std
    coef = np.linalg.solve(xx.T @ (weights[:, None]*xx)+regularization*np.eye(x.shape[1]),
                           xx.T @ (weights*(target-y)))
    return mean, std, y, coef


def predict(train, rows):
    models = {c: fit(train, c) for c in {l for l, _ in CALIBRATED.values()}}
    predictions = []
    for r in rows:
        actions = {name: choose_direct(r['proposals'], r['votes'], p) for name, p in POLICIES.items()}
        for name, (shrinkage, threshold) in CALIBRATED.items():
            mean, std, y, coef = models[shrinkage]
            scores = ((np.asarray(r['features'])-mean)/std) @ coef + y
            best = int(np.argmax(scores))
            actions[name] = best if scores[best] > threshold else None
        score = {name: r['current'] + (r['gains'][action] if action is not None else 0.)
                 for name, action in actions.items()}
        ranks = {name: r['reference'] if action is None else r['proposals'][action]['ranking']
                 for name, action in actions.items()}
        predictions.append({k: r[k] for k in ('domain', 'qid', 'actual_reference', 'cohort')} |
                           {'scores': score, 'rankings': ranks, 'actions': actions})
    return predictions


def choose(rows):
    return max(SETTINGS, key=lambda a: macro((r['domain'], r['scores'][a]) for r in rows))


def crossfit(rows):
    fold_ids = sorted({r['cohort'] for r in rows})
    outer, choices = [], {}
    for held in fold_ids:
        train = [r for r in rows if r['cohort'] != held]
        test = [r for r in rows if r['cohort'] == held]
        inner = []
        for f in fold_ids:
            if f != held:
                inner.extend(predict([r for r in train if r['cohort'] != f],
                                     [r for r in train if r['cohort'] == f]))
        choices[held] = choose(inner)
        outer.extend(predict(train, test))
    return outer, choices


def summarize(rows, choices):
    policies = {name: [name]*len(rows) for name in SETTINGS}
    policies['selected'] = [choices[r['cohort']] for r in rows]
    return {name: {'score': macro((r['domain'], r['scores'][a]) for r, a in zip(rows, arms, strict=True)),
                   'change': paired_delta([(r['domain'], r['scores'][a]-r['scores']['current'])
                                           for r, a in zip(rows, arms, strict=True)]),
                   'edits': sum(r['actions'][a] is not None for r, a in zip(rows, arms, strict=True))}
            for name, arms in policies.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('train-manifest', 'dev-manifest', 'data-root', 'output', 'selection'):
        ap.add_argument(f'--{name}', type=Path, required=name in ('train-manifest', 'data-root', 'output'))
    ap.add_argument('--train-scores', type=Path, nargs='+', required=True)
    ap.add_argument('--dev-scores', type=Path, nargs='+')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a fresh output path')
    train = prepare(args.train_manifest, args.train_scores, args.data_root, 'train')
    if len(train) != 186:
        raise ValueError('Expected 186 training-fold queries')
    if args.selection:
        original = load(args.selection)
        selected = original['selected']
        new = prepare(args.dev_manifest, args.dev_scores, args.data_root, 'dev')
        rows = predict(train, new)
        if len(rows) != 519:
            raise ValueError('Expected all official dev queries')
        result = {'selected': selected, 'dev': summarize(rows, {'dev': selected}), 'per_query': rows}
        if abs(result['dev']['current']['score'] - .3958942538410169) > 1e-12:
            raise ValueError('Official baseline changed')
    else:
        rows, choices = crossfit(train)
        selected = choose(rows)
        result = {'selected': selected, 'by_fold': choices, 'train': summarize(rows, choices),
                  'exact65': summarize([r for r in rows if r['actual_reference']], choices), 'per_query': rows}
    print(selected, {s: {k: (v['score'], v['edits']) for k, v in result[s].items()}
                     for s in ('train', 'dev', 'exact65') if s in result}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
