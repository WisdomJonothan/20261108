"""Replay head-first six-pair judging with challenger diversity within the cache."""
import argparse
import json
from pathlib import Path

from replay_position_discount import bootstrap, eligible


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def diversify(ordered, budget):
    seen = set()
    first, duplicates = [], []
    for action in ordered:
        doc = action['challenger']
        (duplicates if doc in seen else first).append(action)
        seen.add(doc)
    return (first + duplicates)[:budget]


def outcome(actions):
    winner = max(eligible(actions), key=lambda a: a['mp'], default=None)
    return winner['delta'] if winner else 0.


def replay(cache_rows, reference):
    results = []
    for row in cache_rows:
        key = row['domain'], row['qid']
        ordered = sorted(row['actions'], key=lambda a: (a['slot'], -a['prior']))
        fixed = ordered[:6]
        distinct = diversify(ordered, 6)
        delta1, delta2 = outcome(fixed), outcome(distinct)
        assert len(fixed) == len(distinct) == 6 and abs(delta1 - reference[key]) < 1e-12
        results.append({'domain': key[0], 'qid': key[1], 'baseline': row['baseline'],
                        'external6_delta': delta1, 'distinct6_delta': delta2,
                        'external6_unique_challengers': len({a['challenger'] for a in fixed}),
                        'distinct6_unique_challengers': len({a['challenger'] for a in distinct}),
                        'external6_oracle': max(0., *(a['delta'] for a in fixed)),
                        'distinct6_oracle': max(0., *(a['delta'] for a in distinct))})
    return results


def summarize(rows):
    summary = {'queries': len(rows)}
    baseline = bootstrap([(r['domain'], r['baseline']) for r in rows])['point']
    for key in ('external6', 'distinct6'):
        delta = bootstrap([(r['domain'], r[f'{key}_delta']) for r in rows])
        oracle = bootstrap([(r['domain'], r[f'{key}_oracle']) for r in rows])['point']
        summary[key] = {'ndcg': baseline + delta['point'], 'delta': delta,
                        'oracle': baseline + oracle,
                        'reachable': sum(r[f'{key}_oracle'] > 1e-12 for r in rows),
                        'mean_unique_challengers': sum(r[f'{key}_unique_challengers']
                                                       for r in rows) / len(rows)}
    summary['distinct_vs_original'] = bootstrap(
        [(r['domain'], r['distinct6_delta'] - r['external6_delta']) for r in rows])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary, records = {}, {}
    for split, sources in [('oof', ('train_f0', 'train_f1', 'train_f2')), ('dev', ('dev',))]:
        original = load(args.root / 'results/analysis' / f'position_oracle_{split}_20261124.json')
        refs = {(r['domain'], r['qid']): r['policies']['external6']['delta'] for r in original['per_query']}
        rows = []
        for source in sources:
            cache = load(args.root / 'results/replacement_v2' / f'expanded3_fixed_{source}.json')
            rows.extend(replay(cache['budgets']['18']['per_query'], refs))
        assert len(rows) == len(refs)
        records[split] = rows
        summary[split] = summarize(rows)
    report = {'design': 'Within the same cached eighteen pairs, take the first action for each '
              'unique challenger in head-first order. Fill unused slots from repeated challengers '
              'in the same order. Judge eligibility and maximum confidence are unchanged. One action '
              'per query at most, six pairs judged. Qrels are used only for evaluation.',
              'summary': summary, 'per_query': records}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
