"""Measure available correction from full-label vetoes on fixed Direct suffix pairs."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_abstention_shortlist import outcome
from eval_joint_swap_support import prepare
from replay_position_discount import bootstrap


def audit(rows):
    results = []
    for row in rows:
        direct = row['arms']['supervised']
        if row['routed']:
            pairs = list(zip(direct[3:], row['treatment']))
            veto_harm = direct[:3] + [a for a, b in pairs if b['gold_outcome'] != 'incumbent']
            require_gain = direct[:3] + [a for a, b in pairs if b['gold_outcome'] == 'challenger']
        else:
            veto_harm = require_gain = direct
        scores = {name: outcome(actions, True)['delta'] for name, actions in (
            ('direct', direct), ('veto_harm', veto_harm), ('require_gain', require_gain))}
        oracle = outcome(direct, True)['accepted_oracle_delta'] if row['routed'] else scores['direct']
        results.append({'domain': row['domain'], 'qid': row['qid'], 'baseline': row['baseline'],
                        'routed': row['routed'], **scores, 'routed_accepted_oracle': oracle})
    return results


def summarize(rows):
    result = {'queries': len(rows), 'routed': sum(r['routed'] for r in rows), 'arms': {}}
    for name in ('direct', 'veto_harm', 'require_gain', 'routed_accepted_oracle'):
        delta = [(r['domain'], r[name] - r['direct']) for r in rows]
        result['arms'][name] = {
            'ndcg': bootstrap([(r['domain'], r['baseline'] + r[name]) for r in rows])['point'],
            'paired_vs_direct': bootstrap(delta),
            'helpful': sum(r[name] > 1e-12 for r in rows),
            'harmful': sum(r[name] < -1e-12 for r in rows),
            'recovered_zeroes': sum(r['baseline'] <= 1e-12 and r[name] > 1e-12 for r in rows),
            'new_zeroes': sum(r['baseline'] > 1e-12 and r['baseline'] + r[name] <= 1e-12 for r in rows),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--direct-reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root or args.root / 'data/track1_tempo'
    evaluation = {}
    for path in args.direct_reports:
        report = load(path)
        rows = audit(prepare(args.root, data_root, report, report['folds']))
        evaluation[path.name] = {'summary': summarize(rows), 'per_query': rows}
    result = {
        'protocol': 'Keep the original Direct prefix and all unrouted decisions. For routed queries, '
                    'veto-harm removes suffix actions where only the incumbent is judged relevant; '
                    'require-gain keeps only suffix actions where only the challenger is judged relevant. '
                    'Both use the same probabilities, ordering rule and acceptance gate after filtering. '
                    'The accepted oracle takes the best nonnegative nDCG change among accepted actions. '
                    'These qrels-dependent results measure headroom.',
        'reports': evaluation,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({k: v['summary'] for k, v in evaluation.items()}, indent=1))


if __name__ == '__main__':
    main()
