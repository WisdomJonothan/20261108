"""Evaluate specialist scores as an additional signal on fixed abstention pools."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from replay_anchor_fusion_calibration import build, fit_pairwise, rank
from replay_shortlist_near_duplicates import greedy

WEIGHTS = (.075, .15, .225, .3)
STATIC = {'baseline': [.7, .3, 0.]}
for w in WEIGHTS:
    STATIC[f'replace_{w:g}'] = [.7, .3-w, w]
    STATIC[f'add_{w:g}'] = [.7, .3, w]
NAMES = (*STATIC, 'pairwise_uniform', 'pairwise_discount')


def connect(rows, specialist, label):
    seen = set()
    for r in rows:
        if r['route'] != 'fallback':
            continue
        key = r['domain'], r['qid']
        new = specialist[label][key]
        if new['scores'].keys() != set(r['docs']) or new['cohort'] != r['cohort']:
            raise ValueError(f'Specialist pool differs for {key}')
        r['features'][:, 2] = z(np.array([new['scores'][d] for d in r['docs']]))
        seen.add(key)
    if seen != specialist[label].keys():
        raise ValueError('Specialist archive and abstentions disagree')


def predict(rows, weights, reference):
    out = []
    for r in rows:
        key = r['domain'], r['qid']
        if r['route'] == 'accepted':
            ranking = reference[key]['rankings']['1']
            rankings = dict.fromkeys(weights, ranking)
        else:
            rankings = {name: rank(r, chosen) for name, chosen in weights.items()}
        out.append({k: r[k] for k in ('domain', 'qid', 'cohort', 'route')} |
                   {'rankings': rankings, 'scores': {a: ndcg10(v, r['gold']) for a, v in rankings.items()}})
    return out


def fit(rows):
    training = [r for r in rows if r['route'] == 'fallback']
    return STATIC | {'pairwise_uniform': fit_pairwise(training, False), 'pairwise_discount': fit_pairwise(training, True)}


def summarize(rows, selected):
    result = {}
    for name in (*NAMES, 'selected'):
        values, differences = [], []
        for r in rows:
            option = selected[r['cohort']] if name == 'selected' else name
            score = r['scores'][option]
            values.append((r['domain'], score))
            differences.append((r['domain'], score-r['scores']['baseline']))
        result[name] = {'ndcg': macro(values), 'versus_baseline': paired_delta(differences)}
    return result


def choose(rows):
    return max(NAMES, key=lambda arm: macro((r['domain'], r['scores'][arm]) for r in rows))


def with_redundancy(rows, reports, weights, choices, split):
    source = {(r['domain'], r['qid']): r for r in reports['per_query'] if r['split'] == split}
    observed = []
    for r in rows:
        previous = source.get((r['domain'], r['qid']))
        if previous is None:
            if r['route'] != 'accepted':
                raise ValueError(f'Missing redundancy record for {r["qid"]}')
            observed.append((r['domain'], 0.))
            continue
        strength = float(reports['chosen'] if split == 'dev' else reports['crossfit'][str(r['cohort'])])
        docs = r['docs']
        positions = {d: i for i, d in enumerate(docs)}
        sim = np.zeros((len(docs), len(docs)))
        for d1, d2, value in previous['near_duplicate_edges']:
            i, j = positions[d1], positions[d2]
            sim[i, j] = sim[j, i] = value
        scores = r['features'] @ weights[r['cohort']][choices[r['cohort']]]
        result = ndcg10(greedy(docs, scores, sim, strength), r['gold'])
        observed.append((r['domain'], result - previous['scores'][str(strength)]))
    change = paired_delta(observed)
    return {'ndcg': reports[split]['selected']['ndcg'] + change['point'],
            'versus_selected_duplicate_penalty': change}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('budgeted', 'protected', 'data-root', 'scores', 'dev-8b', 'oof-frozen', 'dev-frozen', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--oof-8b', type=Path, nargs=2, required=True)
    parser.add_argument('--redundancy', type=Path)
    args = parser.parse_args()
    archive, protected = load(args.budgeted), load(args.protected)
    specialist = {}
    data = load(args.scores)
    for split in ('train', 'dev'):
        specialist[split] = index({'per_query': [r for r in data['per_query'] if r['split'] == split]})
    train = build(archive, args.oof_8b, args.oof_frozen, args.data_root, 'oof')
    dev = build(archive, [args.dev_8b], args.dev_frozen, args.data_root, 'dev')
    connect(train, specialist, 'train')
    connect(dev, specialist, 'dev')
    train_heads, dev_heads = (index(protected[name]) for name in ('oof', 'dev'))
    weights_by_fold = {}
    chosen_by_fold = {}
    fitted = []
    for i in (0, 1):
        calib = [r for r in train if r['cohort'] != i]
        held = [r for r in train if r['cohort'] == i]
        weights = fit(calib)
        chosen_by_fold[i] = choose(predict(calib, weights, train_heads))
        weights_by_fold[i] = weights
        fitted.extend(predict(held, weights, train_heads))
    final = fit(train)
    pooled = choose(predict(train, final, train_heads))
    out = predict(dev, final, dev_heads)
    summary = {'choice_by_held_cohort': chosen_by_fold, 'pooled_choice': pooled,
               'weights_by_held_cohort': weights_by_fold, 'pooled_weights': final,
               'oof': summarize(fitted, chosen_by_fold), 'dev': summarize(out, {0: pooled}),
               'oof_per_query': fitted, 'dev_per_query': out}
    for split in ('oof', 'dev'):
        prior = protected[split]['full_policy']['1']['ndcg']
        if abs(summary[split]['baseline']['ndcg']-prior) > 1e-12:
            raise ValueError(f'Guarded baseline differs on {split}')
        for arm, r in summary[split].items():
            change = r['versus_baseline']
            print(split, arm, r['ndcg'], change['point'], change['ci95'], flush=True)
    if args.redundancy:
        duplicate = load(args.redundancy)
        summary['with_redundancy'] = {
            'oof': with_redundancy(train, duplicate, weights_by_fold, chosen_by_fold, 'oof'),
            'dev': with_redundancy(dev, duplicate, {0: final}, {0: pooled}, 'dev'),
        }
        print('Combined with redundancy:', summary['with_redundancy'])
    print('Choices', pooled, chosen_by_fold)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(summary, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
