"""Test robust score transformations with fixed shortlist membership and budget."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index
from replay_anchor_fusion_calibration import build, score

ARMS = ('fixed', 'clip1.5', 'clip2.5', 'penalty0.1', 'penalty0.2', 'reward0.1', 'reward0.2')


def replay(rows, guard):
    results = []
    for row in rows:
        frozen, judge = row['features'][:, :2].T
        baseline = .7 * frozen + .3 * judge
        scores = {'fixed': baseline}
        for level in (1.5, 2.5):
            scores[f'clip{level}'] = .7 * frozen + .3 * np.clip(judge, -level, level)
        for strength in (.1, .2):
            scores[f'penalty{strength}'] = baseline - strength * np.abs(judge-frozen)
            scores[f'reward{strength}'] = baseline + strength * np.maximum(judge-frozen, 0.)
        ranks = {}
        for arm, values in scores.items():
            if row['route'] == 'accepted':
                order = guard[row['domain'], row['qid']]['rankings']['1']
            else:
                order = [row['docs'][i] for i in np.argsort(-values, kind='stable')[:10]]
            ranks[arm] = order
        results.append({k: row[k] for k in ('qid', 'domain', 'cohort')} |
                       {'ranking': ranks, 'scores': {name: score(row, docs) for name, docs in ranks.items()}})
    return results


def choose(rows):
    return max(ARMS, key=lambda k: macro((r['domain'], r['scores'][k]) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('budgeted', 'protected', 'data-root', 'dev-8b', 'oof-frozen', 'dev-frozen', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--oof-8b', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    archive, protected = load(args.budgeted), load(args.protected)
    train = build(archive, args.oof_8b, args.oof_frozen, args.data_root, 'oof')
    dev = build(archive, [args.dev_8b], args.dev_frozen, args.data_root, 'dev')
    train, dev = (replay(rows, index(protected[split])) for rows, split in ((train, 'oof'), (dev, 'dev')))
    choices = {i: choose([r for r in train if r['cohort'] != i]) for i in (0, 1)}
    pooled = choose(train)
    output = {'crossfit': choices, 'pooled': pooled, 'oof_per_query': train, 'dev_per_query': dev}
    for split, rows, chosen in (('oof', train, choices), ('dev', dev, {0: pooled})):
        starting = macro((r['domain'], r['scores']['fixed']) for r in rows)
        if abs(starting-protected[split]['full_policy']['1']['ndcg']) > 1e-12:
            raise ValueError('Unexpected baseline score')
        results = {}
        for label in (*ARMS, 'selected'):
            diffs = [(r['domain'], r['scores'][chosen[r['cohort']] if label == 'selected' else label] -
                      r['scores']['fixed']) for r in rows]
            p = paired_delta(diffs)
            results[label] = {'ndcg': starting + p['point'], 'change': p}
            print(split, label, results[label]['ndcg'], p['ci95'], flush=True)
        output[split] = results
    print('Selected', pooled, choices)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(output, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
