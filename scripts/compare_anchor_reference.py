"""Compare anchor scores and six-pair decisions on the exact same queries."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from replay_position_discount import bootstrap


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare(rows):
    policies = ('frozen_base', 'external6', 'anchor_only', 'anchor_fusion')
    scores = {key: bootstrap([(r['domain'], r[key]) for r in rows])['point'] for key in policies}
    change = [(r['domain'], r['anchor_fusion'] - r['external6']) for r in rows]
    domain_diffs = defaultdict(list)
    for domain, difference in change:
        domain_diffs[domain].append(difference)
    means = {domain: mean(values) for domain, values in domain_diffs.items()}
    values = np.array(list(means.values()))
    rng = np.random.default_rng(20261124)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {'queries': len(rows), 'scores': scores, 'anchor_fusion_vs_external6': bootstrap(change),
            'domain_cluster_bootstrap_95': np.quantile(samples, [.025, .975]).tolist(),
            'improved_domains': sum(v > 1e-12 for v in means.values()),
            'worse_domains': sum(v < -1e-12 for v in means.values()), 'per_domain_delta': means}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = {(r['domain'], r['qid']): r for r in load(args.reference)['per_query']}
    samples, all_rows, seen = {}, [], set()
    for path in args.inputs:
        sample = []
        for row in load(path)['per_query']:
            key = row['domain'], row['qid']
            assert key not in seen
            seen.add(key)
            base = row['arms']['base']['ndcg']
            control = old[key]
            assert abs(base - control['baseline']) < 1e-10
            sample.append({'domain': key[0], 'qid': key[1], 'frozen_base': base,
                           'external6': base + control['policies']['external6']['delta'],
                           'anchor_only': row['arms']['anchor_only']['ndcg'],
                           'anchor_fusion': row['arms']['base_plus_anchor30']['ndcg']})
        samples[path.name] = compare(sample)
        all_rows.extend(sample)
    result = {'scope': 'Matched training-fold samples; anchor fusion uses 198 prompts per query '
              'versus 12 for the reference action policy.',
              'samples': samples, 'combined': compare(all_rows), 'per_query': all_rows}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({'samples': samples, 'combined': result['combined']}, indent=1))


if __name__ == '__main__':
    main()
