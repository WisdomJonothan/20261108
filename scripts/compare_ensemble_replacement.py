"""Compare fresh six-pair judgments on the ensemble and reference rankings."""
import argparse
import json
from pathlib import Path

from eval_position_repair import read_json
from replay_position_discount import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path('results/analysis')
    parser.add_argument('--analysis', type=Path, default=default)
    args = parser.parse_args()
    folder = args.analysis
    base = read_json(folder / 'position_oracle_dev_20261124.json')
    fresh = read_json(folder / 'ce_seed_ensemble_fresh_replacement_dev_20261124.json')
    before = {(r['domain'], r['qid']): r for r in base['per_query']}
    after = fresh['budgets']['6']['per_query']
    assert len(before) == len(after) == fresh['n_queries'] == 519 and fresh['prompts'] == 519 * 12
    assert {(r['domain'], r['qid']) for r in after} == set(before)
    tradeoff, effect, base_difference = [], [], []
    for row in after:
        old = before[row['domain'], row['qid']]
        original_top10 = old['baseline']
        original_gain = old['policies']['external6']['delta']
        tradeoff.append((row['domain'], row['policy'] - original_top10 - original_gain))
        effect.append((row['domain'], row['delta'] - original_gain))
        base_difference.append((row['domain'], row['baseline'] - original_top10))
    scores = fresh['budgets']['6']
    output = {'queries': len(after), 'prompts': fresh['prompts'],
              'scores': {'old_base': base['summary']['baseline'],
                         'old_with_replacement': base['summary']['arms']['external6']['ndcg'],
                         'ensemble_base': scores['macro_baseline'],
                         'ensemble_with_fresh_replacement': scores['macro_policy']},
              'fresh_swap_counts': scores['swap_counts'],
              'paired_diff_new_vs_old': bootstrap(tradeoff),
              'paired_diff_base_only': bootstrap(base_difference),
              'paired_diff_of_replacement_gains': bootstrap(effect)}
    with (folder / 'ce_seed_ensemble_fresh_comparison_20261124.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=1)
    print(json.dumps(output, indent=1))


if __name__ == '__main__':
    main()
