"""Combine the recorded expansion routes with exact edit-budget ceilings."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


ARMS = {
    'fixed_one': ('fixed_positions', '1'),
    'fixed_two': ('fixed_positions', '2'),
    'reorder_one': ('full_reorder', '1'),
    'unrestricted': ('full_reorder', '10'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ceilings', type=Path, required=True)
    parser.add_argument('--routing', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    oracle = index(load(args.ceilings))
    routes = index(load(args.routing))
    if oracle.keys() != routes.keys() or len(oracle) != 519:
        raise ValueError('The edit oracle and routing audit must cover the same 519 queries')
    for key, row in oracle.items():
        if abs(row['baseline'] - routes[key]['competition_score']) > 1e-12:
            raise ValueError(f'Competition score differs across inputs for {key}')

    names = tuple(next(iter(routes.values()))['expanded'])
    result = {name: {} for name in names}
    for band, (mode, edits) in ARMS.items():
        base, room = {}, {}
        for key, row in oracle.items():
            d100, d300 = (row['pools'][str(depth)][mode][edits] for depth in (100, 300))
            base[key], room[key] = d100, d300 - d100
            if room[key] < -1e-12:
                raise ValueError(f'Ideal gain must be nonnegative for {key}')
        baseline = paired_delta([(d, score) for (d, _), score in base.items()])['point']
        for name in names:
            differences = [(d, room[d, q] * r['expanded'][name]) for (d, q), r in routes.items()]
            gain = paired_delta(differences)
            result[name][band] = {'ideal_ndcg': baseline + gain['point'], 'gain_vs_depth100': gain}
    report = {
        'protocol': 'Combine the same query-level depth-100 and depth-300 edit-budget ceilings '
                    'using the saved expansion allocations and code. These are ideal choices '
                    'with the official qrels. Uniform allocation gives a probability-weighted expectation.',
        'queries': len(oracle), 'policies': result,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
        stream.write('\n')
    for name, bands in result.items():
        print(name, {b: round(r['ideal_ndcg'], 6) for b, r in bands.items()})


if __name__ == '__main__':
    main()
