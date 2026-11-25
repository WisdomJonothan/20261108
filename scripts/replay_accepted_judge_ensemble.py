"""Test complementary cached 4B and 8B estimates with rank one protected."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from replay_anchor_fusion_calibration import build, rank, score, weight

BETAS = (0., .1, .2, .3)


def evaluate(rows, smaller):
    results = []
    for r in rows:
        key = r['domain'], r['qid']
        scores = dict.fromkeys(BETAS, 0.)
        rankings = {}
        if r['route'] == 'accepted':
            rec = smaller[key]
            sm = rec.get('judge_scores', rec.get('scores', {}))
            docs = r['docs']
            if not set(docs) <= sm.keys():
                raise ValueError(f'Incomplete cached scores for {key}')
            first = rank(r, weight(.3))[0]
            small = z(np.array([sm[d] for d in docs]))
            a, b = r['features'][:, :2].T
            for beta in BETAS:
                value = .4 * a + (.6 - beta) * b + beta * small
                ranking = [first] + [docs[i] for i in np.argsort(-value, kind='stable') if docs[i] != first]
                rankings[beta] = ranking
                scores[beta] = score(r, ranking)
            scores = {beta: value - scores[0.] for beta, value in scores.items()}
        results.append({k: r[k] for k in ('domain', 'qid', 'route', 'cohort')} |
                       {'scores': scores, 'rankings': rankings})
    return results


def select(rows):
    return max(BETAS, key=lambda beta: macro((r['domain'], r['scores'][beta]) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('budgeted', 'protected', 'data-root', 'dev-8b', 'dev-4b', 'oof-frozen', 'dev-frozen', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('oof-8b', 'oof-4b'):
        parser.add_argument(f'--{name}', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    archive, prior = load(args.budgeted), load(args.protected)
    report = {}
    for name, strong, small, frozen in (
        ('oof', args.oof_8b, args.oof_4b, args.oof_frozen),
        ('dev', [args.dev_8b], [args.dev_4b], args.dev_frozen),
    ):
        rows = build(archive, strong, frozen, args.data_root, name)
        smaller = {}
        for path in small:
            smaller.update(index(load(path)))
        actual = evaluate(rows, smaller)
        report[f'{name}_per_query'] = actual
        base = prior[name]['full_policy']['1']['ndcg']
        if name == 'oof':
            chosen = select(actual)
            cross = {i: select([r for r in actual if r['cohort'] != i]) for i in (0, 1)}
            report['train_selected'] = chosen
            report['crossfit'] = cross
        report[name] = {}
        for label in [*BETAS, 'selected']:
            effects = []
            for r in actual:
                beta = (cross[r['cohort']] if name == 'oof' else chosen) if label == 'selected' else label
                effects.append((r['domain'], r['scores'][beta]))
            delta = paired_delta(effects)
            report[name][label] = {'ndcg': base + delta['point'], 'change': delta}
            print(name, label, base + delta['point'], delta['ci95'], flush=True)
    print('training choices', chosen, cross)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, separators=(',', ':'))
        stream.write('\n')


if __name__ == '__main__':
    main()
