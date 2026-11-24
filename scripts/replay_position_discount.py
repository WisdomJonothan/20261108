"""Compare confidence and a rank-discount priority on fixed eligible actions."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def impact(action):
    discount = 1 / math.log2(action['slot'] + 1)
    if action['type'] == 'internal':
        discount -= 1 / math.log2(action['other_slot'] + 1)
    return discount * (2 * action['mp'] - 1)


def eligible(actions):
    return [a for a in actions if a['agree'] and a['mp'] > .5]


def bootstrap(rows, draws=10000):
    groups = defaultdict(list)
    for domain, delta in rows:
        groups[domain].append(delta)
    rng = np.random.default_rng(20261124)
    samples = np.zeros(draws)
    for domain in sorted(groups):
        values = np.asarray(groups[domain])
        samples += values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return {'point': float(np.mean([np.mean(values) for values in groups.values()])),
            'ci95': np.quantile(samples / len(groups), [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding='utf-8'))
    results = {}
    for arm in ('external6', 'internal6', 'mixed3x3'):
        available, decisions, change = [], [], []
        failures, counts = Counter(), Counter()
        for row in report['per_query']:
            cached = row['actions']
            actions = (cached['external'][:3] + cached['internal'][:3] if arm == 'mixed3x3'
                       else cached['external' if arm == 'external6' else 'internal'])
            choices = eligible(actions)
            best = max(choices, key=impact, default=None)
            delta = best['delta'] if best else 0.
            original = row['policies'][arm]['delta']
            domain = row['domain']
            available.append((domain, delta))
            decisions.append((domain, delta - row['policies']['external6']['delta']))
            change.append((domain, delta - original))
            effect = ('abstained' if best is None else 'helpful' if delta > 1e-12
                      else 'harmful' if delta < -1e-12 else 'neutral')
            counts[effect] += 1
            if row['policies'][arm]['max_delta'] > 1e-12:
                failures['oracle_positive'] += 1
                if not any(a['delta'] > 1e-12 for a in choices):
                    failures['no_beneficial_action_accepted'] += 1
                elif delta <= 1e-12:
                    failures['beneficial_action_accepted_but_not_chosen'] += 1
        relative = bootstrap(available)
        results[arm] = {'ndcg': report['summary']['baseline'] + relative['point'],
                        'change_vs_original_ranking': relative,
                        'change_vs_external6_confidence': bootstrap(decisions),
                        'change_vs_own_confidence': bootstrap(change),
                        'counts': counts, 'failures': failures}
    output = {'folds': report['folds'],
              'protocol': 'Keep the same accepted pairs. Choose the positive action with maximum '
              'discount at the edited slot (or discount difference for an exchange) '
              'times 2p-1. The factor is a fixed priority using the mean two-order '
              'challenger probability and has no fitted coefficients.',
              'results': results}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
