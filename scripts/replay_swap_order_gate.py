"""Test whether original one-order preferences can be accepted safely."""
import argparse
import json
from collections import Counter
from pathlib import Path

from compare_anchor_reference import load
from eval_abstention_shortlist import outcome
from eval_joint_swap_support import prepare
from replay_position_discount import bootstrap, eligible


def replay(rows):
    conditions = {}
    evaluated = []
    for row in rows:
        direct = row['arms']['supervised']
        suffix = direct[3:]
        if row['routed']:
            for action, reduced in zip(suffix, row['treatment']):
                group = conditions.setdefault(reduced['gold_outcome'], Counter())
                orders = int(action['p_fwd'] > .5) + int(action['p_rev'] > .5)
                group[f'{orders}_orders_above_half'] += 1
                group['direct_accepts'] += bool(eligible([action]))
                group['mean_accepts'] += action['mp'] > .5
            alternatives = direct[:3] + [action | {'agree': True} for action in suffix]
        else:
            alternatives = direct
        effects = {name: outcome(actions, True)['delta']
                   for name, actions in (('direct', direct), ('mean_only', alternatives))}
        evaluated.append({'domain': row['domain'], 'qid': row['qid'], 'baseline': row['baseline'],
                          'external6_delta': row['external_delta'], **effects})
    return evaluated, conditions


def summarize(rows):
    result = {'queries': len(rows), 'arms': {}}
    for name in ('external6_delta', 'direct', 'mean_only'):
        result['arms'][name] = {
            'ndcg': bootstrap([(r['domain'], r['baseline'] + r[name]) for r in rows])['point'],
            'helpful': sum(r[name] > 1e-12 for r in rows),
            'harmful': sum(r[name] < -1e-12 for r in rows),
            'recovered_fused_zeroes': sum(r['baseline'] <= 1e-12 and r[name] > 1e-12 for r in rows),
            'new_zeroes': sum(r['baseline'] > 1e-12 and r['baseline'] + r[name] <= 1e-12 for r in rows),
        }
    result['mean_only_vs_direct'] = bootstrap([(r['domain'], r['mean_only'] - r['direct']) for r in rows])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--direct-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root or args.root / 'data/track1_tempo'
    reports = {}
    for path in args.direct_reports:
        control = load(path)
        rows, conditions = replay(prepare(args.root, data_root, control, control['folds']))
        reports[path.name] = {'summary': summarize(rows), 'suffix_pairs_by_qrels': conditions, 'per_query': rows}
    report = {
        'protocol': 'Replay the supervised-shortlist Direct arm with fixed prefix and pair set. '
                    'On routed suffix comparisons only, remove the two-order argmax requirement '
                    'and keep the mean challenger probability threshold above 0.5. Choose the '
                    'same maximum-mean accepted candidate. Qrels score outcomes only.',
        'reports': reports,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps({path: row['summary'] for path, row in reports.items()}, indent=1))


if __name__ == '__main__':
    main()
