"""Trace whether shared-reference fallback can reach each remaining zero query."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from compare_anchor_reference import load
from replay_anchor_fallback import index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=Path('results/analysis'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.analysis
    zeroes = load(data / 'competition_zeroes_20261124.json')['zero_queries']
    anchor = index(load(data / 'anchor_cap20_official_dev_20261124.json'))
    confirmation = index(load(data / 'anchor_head_confirmation_dev_20261124.json'))
    counters = defaultdict(Counter)
    rows = []
    for zero in zeroes:
        key = zero['domain'], zero['qid']
        probe, head = anchor[key], confirmation[key]
        if head['external6'] > 1e-12:
            raise ValueError(f'Expected zero score on the competition run for {key}')
        in_pool = probe['pool_positive'] > 0
        released = head['abstained']
        flags = {
            'competition_zeroes': True,
            'anchor_pool_contains_positive': in_pool,
            'released_to_fallback': released,
            'pool_positive_and_released': in_pool and released,
            'pool_positive_but_blocked': in_pool and not released,
            'fixed_head_recovers': head['protected'] > 1e-12,
            'confirmed_recovers': head['confirm_in_both_orders'] > 1e-12,
            'unprotected_recovers': head['unprotected'] > 1e-12,
        }
        for recovery in ('fixed_head_recovers', 'confirmed_recovers', 'unprotected_recovers'):
            if flags[recovery] and not flags['pool_positive_and_released']:
                raise ValueError(f'Unexpected recovery without access to a positive for {key}')
        counters[zero['bucket']].update({name: int(value) for name, value in flags.items()})
        rows.append(zero | flags)
    result = {
        'scope': 'Official dev description using the competition zero-score queries and scores from '
                 'the fixed shared-reference and confirmation caches. Qrels define the groups.',
        'by_dense_first_positive_rank': counters,
        'per_query': rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(counters, indent=1))


if __name__ == '__main__':
    main()
