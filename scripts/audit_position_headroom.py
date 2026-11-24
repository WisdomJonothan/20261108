"""Decompose the fixed-budget oracle gap into acceptance and selection losses."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from replay_position_discount import eligible


def audit(report):
    summaries = {}
    for arm, result in report['summary']['arms'].items():
        groups = defaultdict(list)
        misses = 0
        for row in report['per_query']:
            cache = row['actions']
            actions = (cache['external'][:3] + cache['internal'][:3] if arm == 'mixed3x3'
                       else cache['external'] if arm == 'external6' else cache['internal'])
            ideal = max([0.] + [a['delta'] for a in actions])
            accepted_ideal = max([0.] + [a['delta'] for a in eligible(actions)])
            observed = row['policies'][arm]['delta']
            assert accepted_ideal <= ideal + 1e-12 and observed <= accepted_ideal + 1e-12
            assert abs(ideal - row['policies'][arm]['max_delta']) < 1e-12
            misses += ideal > 1e-12 and accepted_ideal <= 1e-12
            groups[row['domain']].append((ideal, accepted_ideal, observed))
        totals = [mean(mean(row[i] for row in group) for group in groups.values())
                  for i in range(3)]
        ideal, accepted_ideal, observed = totals
        assert abs(ideal - (result['oracle'] - report['summary']['baseline'])) < 1e-10
        assert abs(observed - result['delta']) < 1e-10
        summaries[arm] = {'oracle_gain': ideal, 'accepted_oracle_gain': accepted_ideal,
                          'observed_gain': observed, 'lost_at_acceptance': ideal - accepted_ideal,
                          'lost_at_selection': accepted_ideal - observed,
                          'queries_with_benefit_but_none_accepted': misses}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=Path('results/analysis'))
    folder = parser.parse_args().analysis
    both = {}
    for split in ('oof', 'dev'):
        path = folder / f'position_repair_{split}_20261124.json'
        both[split] = audit(json.loads(path.read_text(encoding='utf-8')))
    report = {'scope': 'Qrels-defined oracle gains conditioned on fixed action sets. '
              'The accepted oracle may abstain. Domains are equally weighted.',
              'summary': both}
    path = folder / 'position_headroom_20261124.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(both, indent=1))


if __name__ == '__main__':
    main()
