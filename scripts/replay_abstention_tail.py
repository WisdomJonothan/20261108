"""Test reordering the abstention tail with the same document membership."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_accepted_weight_changes import protect
from audit_anchor_dev import macro
from compare_anchor_reference import load
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index
from replay_anchor_fusion_calibration import build, rank, score, weight

OPTIONS = ('baseline', *((k, w) for k in (1, 3) for w in (.0, .15, .45, .6)))


def score_all(rows, accepted):
    result = []
    for r in rows:
        key = r['domain'], r['qid']
        baseline = (rank(r, weight(.3)) if r['route'] == 'fallback'
                    else accepted[key]['rankings']['1'])
        orders = {'baseline': baseline}
        if r['route'] == 'fallback':
            for option in OPTIONS[1:]:
                k, w = option
                docs = r['docs']
                values = r['features'] @ weight(w)
                restricted = [docs[i] for i in np.argsort(-values, kind='stable') if docs[i] in baseline]
                order = protect(baseline, restricted, k)
                orders[str(option)] = order
        else:
            orders.update({str(option): baseline for option in OPTIONS[1:]})
        if any(set(order) != set(baseline) for order in orders.values()):
            raise ValueError(f'Membership differs for {key}')
        result.append({k: r[k] for k in ('domain', 'qid', 'cohort', 'route')} |
                      {'ranks': orders, 'scores': {k: score(r, v) for k, v in orders.items()}})
    return result


def select(rows):
    return max(map(str, OPTIONS), key=lambda k: macro((r['domain'], r['scores'][k]) for r in rows))


def summarize(rows, selected):
    result = {}
    for arm in (*map(str, OPTIONS), 'selected'):
        values, differences = [], []
        for r in rows:
            name = selected[r['cohort']] if arm == 'selected' else arm
            s = r['scores'][name]
            values.append((r['domain'], s))
            differences.append((r['domain'], s-r['scores']['baseline']))
        result[arm] = {'ndcg': macro(values), 'change': paired_delta(differences)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('budgeted', 'protected', 'data-root', 'dev-8b', 'oof-frozen', 'dev-frozen', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--oof-8b', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    archive, protected = load(args.budgeted), load(args.protected)
    train = build(archive, args.oof_8b, args.oof_frozen, args.data_root, 'oof')
    dev = build(archive, [args.dev_8b], args.dev_frozen, args.data_root, 'dev')
    train, dev = (score_all(rows, index(protected[split])) for rows, split in ((train, 'oof'), (dev, 'dev')))
    crossfit = {i: select([r for r in train if r['cohort'] != i]) for i in (0, 1)}
    pooled = select(train)
    result = {'crossfit': crossfit, 'pooled': pooled, 'oof': summarize(train, crossfit),
              'dev': summarize(dev, {0: pooled}), 'oof_per_query': train, 'dev_per_query': dev}
    for split in ('oof', 'dev'):
        expected = protected[split]['full_policy']['1']['ndcg']
        if abs(result[split]['baseline']['ndcg']-expected) > 1e-12:
            raise ValueError('Reference policy differs')
        for arm, r in result[split].items():
            print(split, arm, r['ndcg'], r['change']['ci95'], flush=True)
    print('selected', pooled, crossfit)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
