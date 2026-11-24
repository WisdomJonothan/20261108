"""Compare anchor scores and six-pair decisions on the exact same queries."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from coverage_cf_bound import read_steps
from replay_position_discount import bootstrap


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def compare(rows):
    policies = ('frozen_base', 'external6', 'anchor_only', 'anchor_fusion')
    if all('cap20_replay' in r for r in rows):
        policies += ('cap20_replay',)
    scores = {key: bootstrap([(r['domain'], r[key]) for r in rows])['point'] for key in policies}
    change = [(r['domain'], r['anchor_fusion'] - r['external6']) for r in rows]
    domain_diffs = defaultdict(list)
    for domain, difference in change:
        domain_diffs[domain].append(difference)
    means = {domain: mean(values) for domain, values in domain_diffs.items()}
    values = np.array(list(means.values()))
    rng = np.random.default_rng(20261124)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    result = {'queries': len(rows), 'scores': scores, 'anchor_fusion_vs_external6': bootstrap(change),
              'domain_cluster_bootstrap_95': np.quantile(samples, [.025, .975]).tolist(),
              'improved_domains': sum(v > 1e-12 for v in means.values()),
              'worse_domains': sum(v < -1e-12 for v in means.values()), 'per_domain_delta': means}
    if 'cap20_replay' in scores:
        result['cap20_vs_external6'] = bootstrap(
            [(r['domain'], r['cap20_replay'] - r['external6']) for r in rows])
    annotated = [r for r in rows if 'coverage' in r]
    if annotated:
        result['step_coverage'] = {key: bootstrap([(r['domain'], r['coverage'][key]) for r in annotated])['point']
                                   for key in policies if key in annotated[0]['coverage']}
        result['step_coverage_annotated_queries'] = len(annotated)
        result['step_coverage_annotated_domains'] = len({r['domain'] for r in annotated})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--caps', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    old = {(r['domain'], r['qid']): r for r in load(args.reference)['per_query']}
    annotation = {}
    caps = {(r['domain'], r['qid']): r for path in args.caps for r in load(path)['per_query']}
    samples, all_rows, seen = {}, [], set()
    for path in args.inputs:
        sample = []
        report = load(path)
        split = report.get('split', 'train')
        for row in report['per_query']:
            key = row['domain'], row['qid']
            assert key not in seen
            seen.add(key)
            base = row['arms']['base']['ndcg']
            control = old[key]
            assert abs(base - control['baseline']) < 1e-10
            entry = {'domain': key[0], 'qid': key[1], 'frozen_base': base,
                     'external6': base + control['policies']['external6']['delta'],
                     'anchor_only': row['arms']['anchor_only']['ndcg'],
                     'anchor_fusion': row['arms']['base_plus_anchor30']['ndcg']}
            if key in caps:
                assert abs(base - caps[key]['base']) < 1e-10
                entry['cap20_replay'] = caps[key]['cap20']
            if args.data_root:
                domain, qid = key
                if (domain, split) not in annotation:
                    annotation[domain, split] = read_steps(
                        args.data_root / domain / f'qrels_steps_{split}.txt')
                if steps := annotation[domain, split].get(qid):
                    initial = row['arms']['base']['ranking']
                    external = initial.copy()
                    action = control['policies']['external6']['chosen']
                    if action:
                        assert external[action['slot'] - 1] == action['incumbent']
                        external[action['slot'] - 1] = action['challenger']
                    rankings = {'frozen_base': initial, 'external6': external,
                                'anchor_only': row['arms']['anchor_only']['ranking'],
                                'anchor_fusion': row['arms']['base_plus_anchor30']['ranking']}
                    entry['coverage'] = {k: mean(bool(set(ranking) & gold) for gold in steps.values())
                                         for k, ranking in rankings.items()}
            sample.append(entry)
        samples[path.name] = compare(sample)
        all_rows.extend(sample)
    assert not caps or set(caps) == seen
    result = {'scope': 'Matched-query comparison. The reference uses 12 prompts. The anchor '
              'method uses 198 on 100 documents or 38 on 20. Replayed cap20 uses cached scores.',
              'samples': samples, 'combined': compare(all_rows), 'per_query': all_rows}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({'samples': samples, 'combined': result['combined']}, indent=1))


if __name__ == '__main__':
    main()
