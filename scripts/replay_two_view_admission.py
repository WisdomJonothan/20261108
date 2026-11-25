"""Confirm individual admissions against a second evidence view."""
import argparse
import json
from pathlib import Path

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_rerank_headroom_study import rank, registers, summary
from replay_anchor_fallback import index


ACTIONS = {
    'temporal': 'near_conservative',
    'dispersed': 'deeper_head_fixed',
}


def confirmation(before, after, second, positive):
    additions = [d for d in after if d not in before]
    removed = [d for d in before if d not in after]
    # Strongest second-view evidence competes against the strongest evicted document.
    entrants = sorted(additions, key=lambda d: -second[d])
    incumbents = sorted(removed, key=lambda d: -second[d])
    blocked = {
        new: old
        for new, old in zip(entrants, incumbents, strict=True)
        if second[new] <= second[old] or (positive and second[new] <= 0.)
    }
    return [blocked.get(doc, doc) for doc in after]


def evaluate(queries, sources, data):
    rows = []
    index_by_query = index({'per_query': queries})
    if set(sources) != set(ACTIONS) or any(r['scores'].keys() != index_by_query.keys() for r in sources.values()):
        raise ValueError('Both evidence views must contain every query exactly once')
    labels = {(d, s): read_qrels(data / d / f'qrels_{s}.txt')
              for d, s in {(r['domain'], r['split']) for r in queries}}
    for key, row in index_by_query.items():
        domain, qid = key
        docs = row['docs']
        scores = {k: r['scores'][key]['scores'] for k, r in sources.items()}
        gold = labels[domain, row['split']][qid]
        if any(s.keys() != set(docs) for s in scores.values()):
            raise ValueError(f'Candidate scoring differs for {key}')
        orders = {'baseline': docs[:10]}
        remaining = {}
        for mode, policy in ACTIONS.items():
            orders[mode] = rank(docs, scores[mode], policy)
            alternative = scores['dispersed' if mode == 'temporal' else 'temporal']
            for rule, positive in (('preference', False), ('affirmative', True)):
                arm = f'{mode}/{rule}'
                orders[arm] = confirmation(orders['baseline'], orders[mode], alternative, positive)
                remaining[arm] = sum(doc not in orders['baseline'] for doc in orders[arm])
        if any(len(order) != 10 or len(set(order)) != 10 for order in orders.values()):
            raise ValueError(f'Reranking produced an invalid top ten for {key}')
        rows.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'actual_reference')} |
                    {'rankings': orders, 'scores': {k: ndcg10(v, gold) for k, v in orders.items()},
                     'confirmed_admissions': remaining})
    return rows


def choose(rows):
    return max(rows[0]['scores'], key=lambda a: macro((r['domain'], r['scores'][a]) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-root', 'output', 'selection'):
        parser.add_argument(f'--{name}', type=Path, required=name != 'selection')
    parser.add_argument('--scores', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    queries = load(args.manifest)['queries']
    rows = evaluate(queries, registers(args.scores), args.data_root)
    splits = {r['split'] for r in queries}
    if args.selection:
        if splits != {'dev'} or len(rows) != 519:
            raise ValueError('The evaluation requires the entire official dev set')
        selected = load(args.selection)['selected']
        selected_by_fold = {'dev': selected}
    else:
        if splits != {'train'}:
            raise ValueError('Calibration requires training-fold queries')
        selected = choose(rows)
        selected_by_fold = {
            held: choose([r for r in rows if r['cohort'] != held])
            for held in sorted({r['cohort'] for r in rows})
        }
    report = {'selected': selected, 'selected_by_fold': selected_by_fold,
              'full': summary(rows, selected_by_fold), 'per_query': rows}
    if args.selection:
        if abs(report['full']['baseline']['score'] - .39269165354596675) > 1e-12:
            raise ValueError('Official baseline changed')
    else:
        report['matched_65'] = summary([r for r in rows if r['actual_reference']], selected_by_fold)
    for k in ('full', 'matched_65'):
        if k in report:
            print(k, selected, {a: v['score'] for a, v in report[k].items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
