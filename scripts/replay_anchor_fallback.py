"""Replay a fixed shared-reference fallback when the six-pair judge abstains."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels, read_steps
from replay_position_discount import bootstrap


def index(report):
    rows = report['per_query']
    indexed = {(row['domain'], row['qid']): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Expected one row per domain and query ID')
    return indexed


def protect_first_three(original, reranked):
    head = original[:3]
    return head + [doc for doc in reranked if doc not in head][:7]


def discounted_changes(before, after, gold):
    old = {doc: rank for rank, doc in enumerate(before, 1) if doc in gold}
    new = {doc: rank for rank, doc in enumerate(after, 1) if doc in gold}
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(gold)) + 1))

    def value(rank):
        return 1 / (math.log2(rank + 1) * ideal)

    rank_effect = [0.] * 10
    for rank in old.values():
        rank_effect[rank - 1] -= value(rank)
    for rank in new.values():
        rank_effect[rank - 1] += value(rank)
    return {
        'rank_delta': rank_effect,
        'added_documents': sum(value(rank) for doc, rank in new.items() if doc not in old),
        'removed_documents': -sum(value(rank) for doc, rank in old.items() if doc not in new),
        'retained_documents': sum(value(rank) - value(old[doc]) for doc, rank in new.items() if doc in old),
    }


def replay(reference, source, rankings, data_root, split, protect_head):
    if not rankings.keys() <= source.keys() or not rankings.keys() <= reference.keys():
        raise ValueError('Anchor and reference query IDs must match the replay')
    labels = {
        domain: (
            read_qrels(data_root / domain / f'qrels_{split}.txt'),
            read_steps(data_root / domain / f'qrels_steps_{split}.txt'),
        )
        for domain in sorted({domain for domain, _ in rankings})
    }
    rows = []
    for (domain, qid), (anchor_ranking, anchor_score) in rankings.items():
        key = domain, qid
        base = source[key]['arms']['base']
        six_pair = reference[key]['policies']['external6']
        original_ranking = base['ranking']
        six_ranking = original_ranking.copy()
        choice = six_pair['chosen']
        if choice is not None:
            slot = choice['slot'] - 1
            if six_ranking[slot] != choice['incumbent'] or choice['challenger'] in six_ranking:
                raise ValueError(f'Six-pair decision disagrees with the original ranking for {key}')
            six_ranking[slot] = choice['challenger']
        fallback = choice is None
        gold, all_steps = labels[domain]
        fallback_ranking = protect_first_three(original_ranking, anchor_ranking) if protect_head else anchor_ranking
        candidate_lists = {
            'frozen_base': original_ranking,
            'external6': six_ranking,
            'anchor_fusion': anchor_ranking,
            'fallback': fallback_ranking if fallback else six_ranking,
        }
        if any(len(ranking) != 10 or len(set(ranking)) != 10 for ranking in candidate_lists.values()):
            raise ValueError(f'Expected ten distinct documents per ranking for {key}')
        scores = {arm: ndcg10(ranking, gold[qid]) for arm, ranking in candidate_lists.items()}
        expected_scores = (base['ndcg'], reference[key]['baseline'],
                           reference[key]['baseline'] + six_pair['delta'], anchor_score)
        actual_scores = (scores['frozen_base'], scores['frozen_base'],
                         scores['external6'], scores['anchor_fusion'])
        if any(abs(actual - expected) > 1e-10 for actual, expected in zip(actual_scores, expected_scores)):
            raise ValueError(f'Cached scores disagree with qrels for {key}')
        relevant_counts = {arm: len(set(ranking) & gold[qid]) for arm, ranking in candidate_lists.items()}
        external_relevant = set(six_ranking) & gold[qid]
        fallback_relevant = set(candidate_lists['fallback']) & gold[qid]
        row = {'domain': domain, 'qid': qid, 'used_fallback': fallback, 'scores': scores,
               'relevant_counts': relevant_counts,
               'relevant_gained': len(fallback_relevant - external_relevant),
               'relevant_lost': len(external_relevant - fallback_relevant),
               'discounted_changes': discounted_changes(six_ranking, candidate_lists['fallback'], gold[qid])}
        if steps := all_steps.get(qid):
            row['step_coverage'] = {
                arm: mean(bool(set(ranking) & step_docs) for step_docs in steps.values())
                for arm, ranking in candidate_lists.items()
            }
        rows.append(row)
    return rows


def compare(rows):
    def macro(values):
        groups = defaultdict(list)
        for domain, value in values:
            groups[domain].append(value)
        return mean(map(mean, groups.values())) if groups else None

    effects = defaultdict(list)
    for row in rows:
        delta = row['scores']['fallback'] - row['scores']['external6']
        effects[row['domain']].append(delta)
    per_domain = {domain: mean(values) for domain, values in sorted(effects.items())}
    domain_means = np.array(list(per_domain.values()))
    rng = np.random.default_rng(20261124)
    domain_bootstrap = domain_means[rng.integers(len(domain_means), size=(10000, len(domain_means)))].mean(axis=1)
    deltas = [delta for values in effects.values() for delta in values]
    fallback_queries = sum(row['used_fallback'] for row in rows)
    arms = rows[0]['scores']
    return {
        'queries': len(rows),
        'fallback_queries': fallback_queries,
        'mean_prompts': 12 + 38 * fallback_queries / len(rows),
        'scores': {arm: macro((row['domain'], row['scores'][arm]) for row in rows) for arm in arms},
        'delta_vs_external6': bootstrap([(domain, delta) for domain, values in effects.items() for delta in values]),
        'domain_cluster_ci95': np.quantile(domain_bootstrap, [.025, .975]).tolist(),
        'per_domain_delta': per_domain,
        'helpful_queries': sum(delta > 1e-12 for delta in deltas),
        'harmful_queries': sum(delta < -1e-12 for delta in deltas),
        'recovered_zeroes': sum(row['scores']['external6'] == 0 < row['scores']['fallback'] for row in rows),
        'new_zeroes': sum(row['scores']['fallback'] == 0 < row['scores']['external6'] for row in rows),
        'relevant_gained': sum(row['relevant_gained'] for row in rows),
        'relevant_lost': sum(row['relevant_lost'] for row in rows),
        'discounted_changes': {
            **{
                key: macro((row['domain'], row['discounted_changes'][key]) for row in rows)
                for key in ('added_documents', 'removed_documents', 'retained_documents')
            },
            'per_rank': [
                macro((row['domain'], row['discounted_changes']['rank_delta'][rank]) for row in rows)
                for rank in range(10)
            ],
        },
        'step_coverage_queries': sum('step_coverage' in row for row in rows),
        'step_coverage': {
            arm: macro((row['domain'], row['step_coverage'][arm]) for row in rows if 'step_coverage' in row)
            for arm in arms
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--cap', type=Path, nargs='*', default=[])
    parser.add_argument('--protect-head', action='store_true',
                        help='Keep the first three original positions on fallback queries')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cap and len(args.input) != len(args.cap):
        parser.error('Each input must have its own corresponding capped replay')
    reference = index(load(args.reference))
    samples, all_rows, seen = {}, [], set()
    for i, path in enumerate(args.input):
        report = load(path)
        split = report.get('split', 'train')
        source = index(report)
        if seen.intersection(source):
            raise ValueError('Evaluation samples must be disjoint')
        seen.update(source)
        if args.cap:
            cap = index(load(args.cap[i]))
            if cap.keys() != source.keys():
                raise ValueError('Capped replay must cover exactly the source query IDs')
            rankings = {key: (row['ranking'], row['cap20']) for key, row in cap.items()}
        else:
            if report['pool_depth'] != 20:
                raise ValueError('Expected a direct 20-document run or its capped replay')
            arms = {key: row['arms']['base_plus_anchor30'] for key, row in source.items()}
            rankings = {key: (arm['ranking'], arm['ndcg']) for key, arm in arms.items()}
        rows = replay(reference, source, rankings, args.data_root, split, args.protect_head)
        samples[path.name] = compare(rows)
        all_rows.extend(rows)
    result = {
        'policy': 'Preserve each accepted external6 replacement. If external6 abstains, '
                  'use the fixed 20-document 70:30 shared-reference fusion. '
                  'The rule uses no qrels and was evaluated after observing anchor-only and '
                  'anchor-fusion results on dev. This is an exploratory comparison.',
        'protected_head_ranks': 3 if args.protect_head else 0,
        'samples': samples,
        'combined': compare(all_rows),
        'per_query': all_rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result['combined'], indent=1))


if __name__ == '__main__':
    main()
