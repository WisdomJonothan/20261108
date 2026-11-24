"""Measure fixed-budget routes from dense depth 100 to depth 300."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from compare_anchor_reference import load
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


def read_inputs(root, ceilings, reference):
    oracle = index(load(ceilings))
    policies = index(load(reference))
    original = {(r['domain'], r['qid']): r['frozen_top10']
                for r in load(root / 'results/replacement_v2/candidates_dev.json')['queries']}
    if oracle.keys() != policies.keys() or oracle.keys() != original.keys() or len(oracle) != 519:
        raise ValueError('Expected the same 519 official dev queries in all inputs')
    groups = defaultdict(list)
    for domain in sorted({domain for domain, _ in oracle}):
        run = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        for (d, qid), row in oracle.items():
            if d != domain:
                continue
            result = policies[d, qid]
            action = result['policies']['external6']
            if abs(result['baseline'] + action['delta'] - row['baseline']) > 1e-12:
                raise ValueError(f'Competition score disagrees for {domain}/{qid}')
            dense = run[qid][:100]
            if len(dense) != 100:
                raise ValueError(f'Missing dense candidates for {domain}/{qid}')
            scores = [score for _, score, _ in dense]
            source = set(original[d, qid])
            pool100, pool300 = (row['pools'][str(n)] for n in (100, 300))
            ready100 = bool(row['relevant_top10'] + pool100['incoming_positives'])
            ready300 = bool(row['relevant_top10'] + pool300['incoming_positives'])
            before, after = (item['full_reorder']['10'] for item in (pool100, pool300))
            if (ready100 and not ready300) or after < before - 1e-12:
                raise ValueError(f'Depth oracles are not nested for {domain}/{qid}')
            groups[domain].append({
                'domain': domain, 'qid': qid, 'abstained': action['chosen'] is None,
                'top_ten_overlap': len(source & {doc for _, _, doc in dense[:10]}),
                'dense_margin': (scores[0] - scores[9]) / max(pstdev(scores), 1e-8),
                'oracle100': before, 'oracle300': after,
                'newly_reachable': ready300 and not ready100,
                'competition_score': row['baseline'],
            })
    return groups


def route(groups):
    arms = ('abstention', 'small_dense_margin', 'rank_disagreement', 'oracle_selection')
    exported = []
    for rows in groups.values():
        capacity = sum(r['abstained'] for r in rows)
        weakest = sorted(rows, key=lambda r: (r['dense_margin'], r['qid']))[:capacity]
        disagreement = sorted(rows, key=lambda r: (r['top_ten_overlap'], r['dense_margin'], r['qid']))[:capacity]
        ideal = sorted(rows, key=lambda r: (r['oracle100'] - r['oracle300'], r['qid']))[:capacity]
        chosen = [{r['qid'] for r in selection} for selection in
                  ([r for r in rows if r['abstained']], weakest, disagreement, ideal)]
        for row in rows:
            row['expanded'] = {name: float(row['qid'] in selected) for name, selected in zip(arms, chosen)}
            row['expanded'].update(matched_random=capacity / len(rows), all_depth300=1.0, all_depth100=0.0)
            exported.append(row)
    return exported


def summarize(rows):
    summary = {}
    for route_name in rows[0]['expanded']:
        ds = [(r['domain'], (r['oracle300'] - r['oracle100']) * r['expanded'][route_name]) for r in rows]
        expansion = sum(r['expanded'][route_name] for r in rows)
        summary[route_name] = {
            'queries_expanded': expansion,
            'avg_documents': 100 + 200 * expansion / len(rows),
            'delta_ideal_ndcg': paired_delta(ds),
            'newly_reachable_zeroes': sum(r['expanded'][route_name] for r in rows if r['newly_reachable']),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ceilings', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    groups = read_inputs(args.root, args.ceilings, args.reference)
    rows = route(groups)
    baseline = mean(mean(r['oracle100'] for r in items) for items in groups.values())
    result = {
        'protocol': 'Preserve the official dense top 100, and expand selected queries to depth 300. '
                    'Every selective policy uses the same number of expansions within each domain as '
                    'the six-pair abstention policy. The oracle selects the largest ideal improvements. '
                    'Uniform selection gives an exact expectation in each domain. These are ideal '
                    'qrels-based ranking scores. The random interval averages expansion probabilities '
                    'per query before the paired bootstrap.',
        'queries': len(rows), 'domains': len(groups), 'depth100_ideal_ndcg': baseline,
        'policies': summarize(rows),
        'rank_disagreement_paired_advantage': {
            control: paired_delta([
                (r['domain'], (r['oracle300'] - r['oracle100']) *
                 (r['expanded']['rank_disagreement'] - r['expanded'][control])) for r in rows
            ]) for control in ('abstention', 'matched_random')
        },
        'per_query': rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, separators=(',', ':'))
        stream.write('\n')
    for name, stat in result['policies'].items():
        print(name, {k: stat[k] for k in ('queries_expanded', 'avg_documents', 'newly_reachable_zeroes')},
              {k: stat['delta_ideal_ndcg'][k] for k in ('point', 'ci95')})


if __name__ == '__main__':
    main()
