"""Preserve the first three comparisons and diversify the last three on abstention."""
import argparse
import json
from collections import Counter
from pathlib import Path

from compare_anchor_reference import load
from replay_anchor_fallback import index
from replay_position_discount import bootstrap, eligible


def diverse_tail(ordered):
    prefix = ordered[:3]
    if eligible(prefix):
        return ordered[3:6]
    seen = {a['challenger'] for a in prefix}
    new, repeated = [], []
    for action in ordered[3:]:
        doc = action['challenger']
        (repeated if doc in seen else new).append(action)
        seen.add(doc)
    return (new + repeated)[:3]


def summarize(rows):
    base = bootstrap([(r['domain'], r['baseline']) for r in rows])['point']
    output = {'queries': len(rows), 'abstained_after_three': sum(r['prefix_abstention'] for r in rows),
              'changed_suffix': sum(r['changed_suffix'] for r in rows), 'arms': {}}
    for name in ('external6', 'adaptive6'):
        changes = [(r['domain'], r[name]['delta']) for r in rows]
        effect = bootstrap(changes)
        counts = Counter('abstained' if r[name]['chosen'] is None else
                         'helpful' if r[name]['delta'] > 1e-12 else
                         'harmful' if r[name]['delta'] < -1e-12 else 'neutral' for r in rows)
        output['arms'][name] = {
            'ndcg': base + effect['point'], 'delta_vs_fused': effect, 'counts': dict(counts),
            'oracle_ndcg': base + bootstrap([(r['domain'], r[name]['oracle']) for r in rows])['point'],
            'mean_unique_challengers': sum(r[name]['distinct'] for r in rows) / len(rows),
            'fused_zero_queries': sum(r['baseline'] <= 1e-12 for r in rows),
            'fused_zeroes_reachable': sum(r['baseline'] <= 1e-12 and r[name]['oracle'] > 1e-12 for r in rows),
            'fused_zeroes_recovered': sum(r['baseline'] <= 1e-12 and r[name]['delta'] > 1e-12 for r in rows),
            'new_zeroes': sum(r['baseline'] > 1e-12 and r['baseline'] + r[name]['delta'] <= 1e-12 for r in rows),
        }
    output['adaptive_vs_external6'] = bootstrap([
        (r['domain'], r['adaptive6']['delta'] - r['external6']['delta']) for r in rows
    ])
    return output


def evaluate(actions):
    accepted = eligible(actions)
    chosen = max(accepted, key=lambda a: a['mp'], default=None)
    return {
        'chosen': chosen,
        'delta': chosen['delta'] if chosen else 0.,
        'oracle': max(0., *(a['delta'] for a in actions)),
        'distinct': len({a['challenger'] for a in actions}),
    }


def replay(root, split):
    oof = split == 'oof'
    reference = index(load(root / 'results/analysis' / f'position_oracle_{split}_20261124.json'))
    sources = ('train_f0', 'train_f1', 'train_f2') if oof else ('dev',)
    rows = []
    for name in sources:
        cache = load(root / 'results/replacement_v2' / f'expanded3_fixed_{name}.json')
        for row in cache['budgets']['18']['per_query']:
            domain, qid = row['domain'], row['qid']
            ref = reference[domain, qid]
            ordered = sorted(row['actions'], key=lambda a: (a['slot'], -a['prior']))
            fixed = ordered[:6]
            tail = diverse_tail(ordered)
            adaptive = ordered[:3] + tail
            assert len(adaptive) == len(fixed) == 6
            assert abs(row['baseline'] - ref['baseline']) < 1e-12
            external = evaluate(fixed)
            assert abs(external['delta'] - ref['policies']['external6']['delta']) < 1e-12
            rows.append({'domain': domain, 'qid': qid, 'fold': name, 'baseline': row['baseline'],
                         'prefix_abstention': not eligible(ordered[:3]), 'changed_suffix': tail != fixed[3:],
                         'external6': external, 'adaptive6': evaluate(adaptive)})
    assert {(r['domain'], r['qid']) for r in rows} == reference.keys() and len(rows) == len(reference)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--split', choices=('oof', 'dev'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = replay(args.root, args.split)
    report = {
        'protocol': 'Use the first three original comparisons. If none passes the fixed two-order gate, '
                    'spend the last three comparisons on challengers not yet seen, in the original '
                    'head-first cache order. Fill unused slots from repeated challengers in that order. '
                    'Keep the original suffix if any of the first three passes. Choose the '
                    'highest-mean-probability accepted action with at most one replacement.',
        'split': args.split,
        'summary': summarize(rows),
        'per_query': rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    main()
