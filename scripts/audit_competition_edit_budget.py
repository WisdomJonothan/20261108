"""Compute exact binary-qrels oracle scores for limited additions and reordering."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from compare_anchor_reference import load
from coverage_cf_bound import DOMAINS, ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_fallback import index


DEPTHS = (20, 100, 300, 1000)
BUDGETS = (0, 1, 2, 3, 5, 10)
DISCOUNTS = [1 / math.log2(rank + 1) for rank in range(1, 11)]


def ceilings(top, pool, gold):
    ideal = sum(DISCOUNTS[:min(10, len(gold))])
    relevant = [d in gold for d in top]
    incoming = len((set(pool) - set(top)) & gold)
    dcg = sum(value for hit, value in zip(relevant, DISCOUNTS) if hit)
    free_slots = [value for hit, value in zip(relevant, DISCOUNTS) if not hit]
    fixed, reorder = {}, {}
    for budget in BUDGETS:
        gained = min(budget, incoming, len(free_slots))
        fixed[budget] = (dcg + sum(free_slots[:gained])) / ideal if ideal else 0.
        reorder[budget] = sum(DISCOUNTS[:sum(relevant) + gained]) / ideal if ideal else 0.
    return {'incoming_positives': incoming, 'fixed_positions': fixed, 'full_reorder': reorder}


def compile_queries(root, data_root):
    manifest = load(root / 'results/replacement_v2/candidates_dev.json')
    plans = {(row['domain'], row['qid']): row['frozen_top10'] for row in manifest['queries']}
    ref = index(load(root / 'results/analysis/position_oracle_dev_20261124.json'))
    if plans.keys() != ref.keys():
        raise ValueError('The two cached Direct artifacts disagree on query IDs')
    rows = []
    for domain in DOMAINS:
        gold = read_qrels(data_root / domain / 'qrels_dev.txt')
        dense = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        for key, original in plans.items():
            if key[0] != domain:
                continue
            qid = key[1]
            policy = ref[key]['policies']['external6']
            top = original.copy()
            if action := policy['chosen']:
                slot = action['slot'] - 1
                if top[slot] != action['incumbent'] or action['challenger'] in top:
                    raise ValueError(f'Mismatched competition action for {key}')
                top[slot] = action['challenger']
            baseline = ndcg10(top, gold[qid])
            if abs(baseline - ref[key]['baseline'] - policy['delta']) > 1e-12:
                raise ValueError(f'Fixed scorer disagrees on {key}')
            delivered = [doc for _, _, doc in dense[qid]]
            if len(set(top)) != 10 or not set(top) <= set(delivered[:100]):
                raise ValueError(f'Competition top ten is outside its supplied pool for {key}')
            candidate_order = list(dict.fromkeys(top + delivered))
            if len(candidate_order) < max(DEPTHS):
                raise ValueError(f'Missing delivered candidates for {key}')
            rows.append({'domain': domain, 'qid': qid, 'baseline': baseline, 'relevant_top10': len(set(top) & gold[qid]),
                         'pools': {depth: ceilings(top, candidate_order[:depth], gold[qid]) for depth in DEPTHS}})
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain']].append(row)
    baseline = mean(mean(item['baseline'] for item in group) for group in groups.values())
    result = {'queries': len(rows), 'domains': len(groups), 'competition_ndcg': baseline, 'pools': {}}
    for depth in DEPTHS:
        per_domain = {}
        for domain, queries in groups.items():
            pool = [r['pools'][depth] for r in queries]
            per_domain[domain] = {
                'queries': len(queries), 'no_positive_in_pool': sum(
                    r['relevant_top10'] + r['pools'][depth]['incoming_positives'] == 0 for r in queries),
                **{arm: {budget: mean(p[arm][budget] for p in pool) for budget in BUDGETS}
                   for arm in ('fixed_positions', 'full_reorder')},
            }
        combined = {
            arm: {budget: mean(d[arm][budget] for d in per_domain.values()) for budget in BUDGETS}
            for arm in ('fixed_positions', 'full_reorder')
        }
        result['pools'][depth] = combined | {'per_domain': per_domain}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = args.data_root or args.root / 'data/track1_tempo'
    rows = compile_queries(args.root, data)
    summary = summarize(rows)
    if len(rows) != 519 or summary['domains'] != 13:
        raise ValueError('Expected the complete official dev set')
    report = {
        'protocol': 'Reconstruct the frozen six-pair competition output. For each budget and depth, '
                    'protect that output in the pool and fill from the delivered dense run. '
                    'With binary qrels, replace the earliest nonpositive slots with up to the '
                    'allowed number of held-back positives. The full-reorder arm sorts the '
                    'same ideal number of positives to the front. These are exact ceilings.',
        'summary': summary, 'per_query': rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    printable = summary | {'pools': {depth: {k: v for k, v in item.items() if k != 'per_domain'}
                                    for depth, item in summary['pools'].items()}}
    print(json.dumps(printable, indent=1))


if __name__ == '__main__':
    main()
