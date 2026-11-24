"""Recompute paired estimates over disjoint shortlist inference reports."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from compare_anchor_reference import load
from eval_abstention_shortlist import summarize
from replay_anchor_fallback import index


def domain_diagnostics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain']].append(row['outcomes']['supervised']['delta'] - row['external_delta'])
    per_domain = {domain: mean(deltas) for domain, deltas in sorted(groups.items())}
    values = np.asarray(list(per_domain.values()))
    rng = np.random.default_rng(20261124)
    indices = rng.integers(len(values), size=(10000, len(values)))
    return {
        'supervised_per_domain_delta_vs_external6': per_domain,
        'supervised_query_weighted_delta_vs_external6': mean(delta for deltas in groups.values() for delta in deltas),
        'supervised_domain_cluster_ci95': np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocols, folds, seen, inference = set(), {}, set(), []
    rows = []
    for path in args.inputs:
        report = load(path)
        if report['inference'] is None:
            raise ValueError(f'{path} is an oracle report without judge predictions')
        indexed = index(report)
        if seen.intersection(indexed) or folds.keys() & report['per_fold'].keys():
            raise ValueError('Inputs must come from separate held-out folds')
        protocols.add(report['protocol'])
        seen.update(indexed)
        rows.extend(indexed.values())
        folds.update(report['per_fold'])
        inference.append(report['inference'])
    if len(protocols) != 1:
        raise ValueError('Inputs must use identical decision protocols')
    summary = summarize(rows, scored=True) | domain_diagnostics(rows)
    prompts = sum(value['prompts'] for value in inference)
    if prompts != 2 * summary['fresh_pairs']:
        raise ValueError('Prompt count differs from two orders per fresh pair')
    result = {
        'protocol': protocols.pop(),
        'inputs': [p.name for p in args.inputs],
        'inference': {
            'prompts': prompts,
            'prompt_tokens': sum(value['prompt_tokens'] for value in inference),
            'judge_seconds': sum(value['seconds_total'] for value in inference),
            'gpu_peak_mib': max(value['gpu_peak_mb'] for value in inference),
        },
        'summary': summary,
        'per_fold': folds,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
