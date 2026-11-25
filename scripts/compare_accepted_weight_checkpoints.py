"""Compare judge-weight sensitivity on identical accepted lists across model sizes."""
import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from replay_anchor_fusion_calibration import build, rank, score, weight


def evaluate(rows, depth100, checkpoints4b):
    strong = {}
    for path in depth100:
        strong.update(index(load(path)))
    smaller = {}
    for path in checkpoints4b:
        cohort = index(load(path))
        if not smaller.keys().isdisjoint(cohort):
            raise ValueError('4B score cohorts overlap')
        smaller.update(cohort)
    accepted = [r for r in rows if r['route'] == 'accepted']
    results = []
    for row in accepted:
        key = row['domain'], row['qid']
        old = smaller[key]
        if not set(row['docs']) <= old['scores'].keys():
            continue
        if old['anchor'] != strong[key]['anchor']:
            raise ValueError(f'Mismatched reference document for {key}')
        features = row['features'].copy()
        features[:, 1] = z(np.array([old['scores'][doc] for doc in row['docs']]))
        scored = {'4b': row | {'features': features}, '8b': row}
        orders = {model: {str(alpha): rank(r, weight(alpha)) for alpha in (.3, .6)}
                  for model, r in scored.items()}
        scores = {model: {alpha: score(row, order) for alpha, order in ranks.items()}
                  for model, ranks in orders.items()}
        results.append({k: row[k] for k in ('domain', 'qid', 'cohort')} |
                       {'rankings': orders, 'ndcg': scores})
    arms = {}
    for name in ('4b', '8b'):
        arms[name] = {alpha: macro((r['domain'], r['ndcg'][name][alpha]) for r in results)
                      for alpha in ('0.3', '0.6')}
        arms[name]['change'] = paired_delta([
            (r['domain'], r['ndcg'][name]['0.6'] - r['ndcg'][name]['0.3']) for r in results
        ])
    contrast = paired_delta([
        (r['domain'], (r['ndcg']['8b']['0.6'] - r['ndcg']['8b']['0.3']) -
         (r['ndcg']['4b']['0.6'] - r['ndcg']['4b']['0.3'])) for r in results
    ])
    return {'accepted_queries': len(accepted), 'matched_queries': len(results),
            'domains': len({r['domain'] for r in results}), 'arms': arms,
            'weight_effect_8b_minus_4b': contrast, 'per_query': results}


def scale_ratio(row):
    return float(row['features'][:, 0].std())


def equivalent_weight(ratio):
    return .3 / (.3 + .7 * ratio)


def scale_equivalence(train, dev):
    train_accepted = [r for r in train if r['route'] == 'accepted']
    pooled = equivalent_weight(median(scale_ratio(r) for r in train_accepted))
    crossfit = {
        cohort: equivalent_weight(median(
            scale_ratio(r) for r in train_accepted if r['cohort'] != cohort
        ))
        for cohort in {r['cohort'] for r in train_accepted}
    }
    report = {'fixed_equivalent_weight_selected_on_training': pooled,
              'crossfit_equivalent_weight': crossfit}
    for split, rows in (('oof', train), ('dev', dev)):
        effects = []
        actual_weights = []
        rank_differences = 0
        for r in rows:
            before = rank(r, weight(.3))
            old = score(r, before)
            fixed_value = nominal_value = old
            if r['route'] == 'accepted':
                ratio = scale_ratio(r)
                relative_weight = equivalent_weight(ratio)
                features = r['features'].copy()
                if ratio > 0:
                    features[:, 0] = (features[:, 0] - features[:, 0].mean()) / ratio
                current = r | {'features': features}
                equivalent = rank(current, weight(relative_weight))
                if equivalent != before:
                    rank_differences += 1
                if abs(score(r, equivalent) - old) > 1e-12:
                    raise ValueError(f'Equivalent weights changed graded positions for {r["qid"]}')
                fixed_value = score(r, rank(current, weight(pooled if split == 'dev' else crossfit[r['cohort']])))
                nominal_value = score(r, rank(current, weight(.3)))
                actual_weights.append(relative_weight)
            effects.append((r['domain'], old, fixed_value, nominal_value))
        report[split] = {
            'accepted_queries': len(actual_weights),
            'equivalent_rankings_with_float_tie_difference': rank_differences,
            'effective_weight_quantiles': np.quantile(actual_weights, [.1, .5, .9]).tolist(),
            'full_aggregate': {
                name: macro((r[0], r[i]) for r in effects)
                for i, name in enumerate(('domain', 'per_query_equivalent', 'fixed_equivalent', 'nominal_local')) if i
            },
            'fixed_versus_per_query': paired_delta([(d, fixed - old) for d, old, fixed, _ in effects]),
            'fixed_versus_nominal_local': paired_delta([(d, fixed - nominal) for d, _, fixed, nominal in effects]),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgeted', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--oof-8b', type=Path, nargs=2, required=True)
    parser.add_argument('--dev-8b', type=Path, required=True)
    parser.add_argument('--oof-4b', type=Path, nargs=2, required=True)
    parser.add_argument('--dev-4b', type=Path, required=True)
    parser.add_argument('--oof-frozen', type=Path, required=True)
    parser.add_argument('--dev-frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    choices = {'oof': (args.oof_8b, args.oof_4b, args.oof_frozen),
               'dev': ([args.dev_8b], [args.dev_4b], args.dev_frozen)}
    archive = load(args.budgeted)
    result = {}
    by_split = {}
    for split, (strong, smaller, frozen) in choices.items():
        rows = build(archive, strong, frozen, args.data_root, split)
        by_split[split] = rows
        result[split] = evaluate(rows, strong, smaller)
        summary = {k: v for k, v in result[split].items() if k != 'per_query'}
        print(split, summary, flush=True)
    result['scale_equivalence'] = scale_equivalence(by_split['oof'], by_split['dev'])
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
