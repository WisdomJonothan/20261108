"""Evaluate two-order agreement for top-three exchanges after external abstention."""
import argparse
import json
from collections import Counter
from pathlib import Path

from compare_anchor_reference import load
from replay_position_discount import bootstrap


def top_exchange(actions, require_agreement):
    return max(
        (action for action in actions if action['mp'] > .5
         and (action['agree'] or not require_agreement)),
        key=lambda action: action['mp'],
        default=None,
    )


def replay(source):
    rows = []
    for query in source['per_query']:
        decision = query['policies']['external6']
        abstained = decision['chosen'] is None
        actions = [action for action in query['actions']['internal'] if action['slot'] <= 3] if abstained else []
        mean_only = top_exchange(actions, require_agreement=False)
        two_orders = top_exchange(actions, require_agreement=True)
        rows.append({
            'domain': query['domain'], 'qid': query['qid'], 'external_abstained': abstained,
            'external6': query['baseline'] + decision['delta'],
            'pairs': len(actions),
            'mean_only': mean_only, 'two_orders': two_orders,
            'oracle_delta': max((action['delta'] for action in actions), default=0.) if actions else 0.,
            'accepted_oracle_delta': max((action['delta'] for action in actions
                                          if action['agree'] and action['mp'] > .5), default=0.),
        })
    return rows


def summarize(rows):
    score = bootstrap([(row['domain'], row['external6']) for row in rows])['point']
    result = {
        'queries': len(rows),
        'external_abstentions': sum(row['external_abstained'] for row in rows),
        'external6': score,
        'mean_prompts': 12 + 2 * sum(row['pairs'] for row in rows) / len(rows),
        'oracle_delta': bootstrap([(row['domain'], max(0., row['oracle_delta'])) for row in rows])['point'],
        'accepted_oracle_delta': bootstrap([
            (row['domain'], max(0., row['accepted_oracle_delta'])) for row in rows
        ])['point'],
        'arms': {},
    }

    def delta(row, arm):
        action = row[arm]
        return action['delta'] if action else 0.

    for arm in ('mean_only', 'two_orders'):
        counts = Counter(
            'no_exchange' if row[arm] is None else
            'helpful' if delta(row, arm) > 1e-12 else
            'harmful' if delta(row, arm) < -1e-12 else 'neutral'
            for row in rows
        )
        effect = bootstrap([(row['domain'], delta(row, arm)) for row in rows])
        result['arms'][arm] = {'ndcg': score + effect['point'], 'delta_vs_external6': effect, 'counts': dict(counts)}
    result['two_orders_vs_mean_only'] = bootstrap([
        (row['domain'], delta(row, 'two_orders') - delta(row, 'mean_only')) for row in rows
    ])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = load(args.input)
    rows = replay(source)
    summary = summarize(rows)
    report = {
        'protocol': 'Keep accepted external6 decisions. Otherwise offer the already proposed '
                    'internal exchanges involving original rank 1, 2 or 3. Assess the '
                    'maximum mean challenger probability above 0.5 with and without the '
                    'requirement that the challenger win in both presentation orders. '
                    'Both policies use the same comparisons and make at most one exchange.',
        'folds': source['folds'],
        'summary': summary,
        'per_query': rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
