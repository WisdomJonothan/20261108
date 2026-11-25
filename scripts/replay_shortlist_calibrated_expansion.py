"""Replay candidate expansion with shortlist-calibrated scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels, read_steps
from replay_anchor_fallback import index
from replay_novel_candidate_pools import prepare, select_queries
from replay_anchor_edit_scope import z
from replay_shortlist_near_duplicates import greedy, similarities
from summarize_novel_candidate_pools import evaluate, fit_proxy


WIDTHS = {'dense34': 34, 'local58': 58, 'calibrated58': 58, 'one58': 58,
          'local100': 100, 'calibrated100': 100, 'one100': 100}
CONTAINMENT = {
    'dense34': (1., 0.), 'contain90': (.90, 1.), 'contain95': (.95, 1.),
    'double90': (.90, 2.), 'double95': (.95, 2.),
}


def scale(scores, reference):
    values = scores[reference]
    return (scores - values.mean()) / max(float(values.std()), 1e-8)


def one_addition(docs, values, penalties, original, first):
    remain = values.copy()
    expanded = np.ones(len(docs), dtype=bool)
    expanded[original] = False
    first_index = docs.index(first)
    remain[first_index] = -np.inf
    maximum = penalties[first_index].copy()
    order = [first]
    for _ in range(9):
        i = int(np.argmax(remain - maximum))
        order.append(docs[i])
        remain[i] = -np.inf
        np.maximum(maximum, penalties[i], out=maximum)
        if expanded[i]:
            remain[expanded] = -np.inf
    return order


def replay(root, data, selected):
    rows = []
    for row, docs, frozen, gold, windows in prepare(root, data, selected):
        start = set(row['frozen_top10'])
        insiders = [i for i, doc in enumerate(docs) if doc in start]
        outsiders = [i for i, doc in enumerate(docs) if doc not in start]
        pools = {width: sorted(insiders + outsiders[:width-10]) for width in set(WIDTHS.values())}
        scores = np.array([frozen[doc] for doc in docs])
        raw = row.get('anchor_scores')
        judge = np.array([raw[doc] for doc in docs]) if raw else np.zeros(len(docs))
        sim = similarities(windows)
        penalties = np.where(sim >= .95, 2 * sim, sim)
        rankings, oracles = {}, {}
        for arm, width in WIDTHS.items():
            membership = pools[width]
            chosen = [docs[i] for i in membership]
            scale_on = membership if arm.startswith('local') else pools[34]
            values = (.7 * scale(scores, scale_on) + .3 * scale(judge, scale_on))[membership]
            pair_penalties = penalties[np.ix_(membership, membership)]
            if arm.startswith('one'):
                positions = {i: j for j, i in enumerate(membership)}
                original = [positions[i] for i in pools[34]]
                order = one_addition(chosen, values, pair_penalties, original, rankings['dense34'][0])
                if len(set(order) - {chosen[i] for i in original}) > 1:
                    raise ValueError(f'Too many admissions in {row["qid"]}')
            else:
                order = greedy(chosen, values, pair_penalties, 1.)
            rankings[arm] = order
            oracles[arm] = ndcg10(sorted(gold & set(chosen)), gold)
        rows.append({key: row[key] for key in ('domain', 'qid', 'sample', 'fold')} |
                    {'rankings': rankings, 'scores': {arm: ndcg10(rank, gold)
                                                     for arm, rank in rankings.items()},
                     'oracle': oracles, 'novel_members_changed': {
                         arm: width - 34 for arm, width in WIDTHS.items()}})
    return rows


def contained_by(windows):
    result = np.zeros((len(windows), len(windows)))
    for i, retained in enumerate(windows):
        for j, candidate in enumerate(windows):
            if len(candidate) >= 12 and .35 * len(retained) <= len(candidate) < .95 * len(retained):
                result[i, j] = len(retained & candidate) / len(candidate)
    return result


def replay_containment(root, data, selected):
    rows = []
    for row, docs, frozen, gold, windows in prepare(root, data, selected, depth=34):
        sim = similarities(windows)
        baseline = np.where(sim >= .95, 2 * sim, sim)
        overlap = contained_by(windows)
        values = .7 * z([frozen[doc] for doc in docs])
        if 'anchor_scores' in row:
            values += .3 * z([row['anchor_scores'][doc] for doc in docs])
        rankings = {}
        for arm, (threshold, strength) in CONTAINMENT.items():
            penalty = np.maximum(baseline, np.where(overlap >= threshold, strength * overlap, 0.))
            rankings[arm] = greedy(docs, values, penalty, 1.)
        scores = {arm: ndcg10(rank, gold) for arm, rank in rankings.items()}
        oracle = ndcg10(sorted(gold & set(docs)), gold)
        rows.append({key: row[key] for key in ('domain', 'qid', 'sample', 'fold')} |
                    {'rankings': rankings, 'scores': scores, 'oracle': dict.fromkeys(CONTAINMENT, oracle),
                     'novel_members_changed': dict.fromkeys(CONTAINMENT, 0)})
    return rows


def estimate(rows, reference, protected, data, split, widths):
    domains = {row['domain'] for row in rows + protected[split]['per_query']}
    source = 'train' if split == 'oof' else 'dev'
    qrels = {domain: read_qrels(data / domain / f'qrels_{source}.txt') for domain in domains}
    steps = ({domain: read_steps(data / domain / 'qrels_steps_dev.txt') for domain in domains}
             if split == 'dev' else None)
    current = {**reference[split], 'per_query': [
        row | {'rankings': {'max_1': row['rankings']['exact_boost_2']}}
        for row in reference[split]['per_query']
    ]}
    results = evaluate(rows, current, protected[split], widths, qrels, steps)
    previous = reference[split]['arms']['exact_boost_2']['ndcg']
    if abs(results['dense34']['ndcg'] - previous) > 1e-12:
        raise ValueError(f'Unexpected aggregated {split} score')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--containment', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    archive = args.root / 'results' / 'analysis'

    def path(stem, date='20261125'):
        return archive / f'{stem}_{date}.json'

    proxy_reference = path('redundancy_stronger_proxy')
    cohorts = [path('anchor_reranker_qwen8b_sample26'), path('anchor_reranker_qwen8b_replication39')]
    selected = select_queries(args.root, proxy_reference, cohorts, path('anchor_reranker_qwen8b_dev100'),
                              path('position_oracle_dev', '20261124'))
    reference = load(path('redundancy_stronger_replay'))
    protected = load(path('accepted_weight_full_dev_audit'))
    run = replay_containment if args.containment else replay
    widths = dict.fromkeys(CONTAINMENT, 34) if args.containment else WIDTHS
    train = run(args.root, args.data_root, [(k, r) for k, r in selected if r['sample'] != 'dev'])
    proxy = [r for r in train if r['sample'] == 'proxy']
    baseline = index(load(proxy_reference))
    if {(r['domain'], r['qid']) for r in proxy} != baseline.keys() or any(
        r['rankings']['dense34'] != baseline[r['domain'], r['qid']]['rankings']['exact_boost_2']
        for r in proxy
    ):
        raise ValueError('Training proxy did not reproduce the published rankings')
    proxy_scores = {arm: macro((r['domain'], r['scores'][arm]) for r in proxy) for arm in widths}
    proxy_choice = max(widths, key=proxy_scores.get)
    oof = [r for r in train if r['sample'] == 'oof']
    evaluation_oof = estimate(oof, reference, protected, args.data_root, 'oof', widths)
    oof_choice = max(widths, key=lambda a: evaluation_oof[a]['ndcg'])
    crossfit = fit_proxy(proxy, widths)
    print('Training-only choices', proxy_choice, oof_choice, crossfit['choice_by_target_fold'], flush=True)

    official = run(args.root, args.data_root, [(k, r) for k, r in selected if r['sample'] == 'dev'])
    evaluation_dev = estimate(official, reference, protected, args.data_root, 'dev', widths)
    result = {'candidate_widths': widths, 'proxy_choice': proxy_choice, 'oof_choice': oof_choice,
              'proxy_scores': proxy_scores, 'crossfit': crossfit, 'oof': evaluation_oof,
              'dev': evaluation_dev, 'per_query': train + official}
    for split in ('oof', 'dev'):
        for arm, report in result[split].items():
            print(split, arm, report['ndcg'], report['change']['ci95'], flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
