"""Replay per-query 8B candidate budgets at equal per-domain prompt count."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median, pstdev

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index

WIDTHS = (12, 18, 24, 30, 48)
SCHEDULES = ((12, 24, 48), (18, 24, 30))
SIGNALS = ('upstream_overlap', 'upstream_margin', 'preview_overlap', 'preview_margin')
NORMS = (('local', 'local'), ('full', 'local'), ('local', 'full'), ('full', 'full'))


def apply(base, anchor, docs, a='local', b='local'):
    frozen, judge = (np.array([source[d] for d in docs]) for source in (base, anchor))
    full_frozen, full_judge = (np.array(list(source.values())) for source in (base, anchor))
    values = .7 * ((frozen - frozen.mean()) / max((frozen if a == 'local' else full_frozen).std(), 1e-8))
    values += .3 * ((judge - judge.mean()) / max((judge if b == 'local' else full_judge).std(), 1e-8))
    order = np.argsort(-values, kind='stable')
    return [docs[i] for i in order[:10]], float(values[order[9]] - values[order[10]])


def build(path, raw_paths, frozen_path, data, split):
    previous = load(path)
    rows = previous[f'{split}_per_query']
    raw = {}
    for path in raw_paths:
        cohort = index(load(path))
        if not raw.keys().isdisjoint(cohort):
            raise ValueError('Score cohorts overlap')
        raw.update(cohort)
    pool_scores = index(load(frozen_path))
    if raw.keys() != {(r['domain'], r['qid']) for r in rows}:
        raise ValueError('Budget report and raw score keys differ')
    gold = {d: read_qrels(data / d / f'qrels_{"train" if split == "oof" else "dev"}.txt') for d, _ in raw}
    result = []
    for row in rows:
        d, q = row['domain'], row['qid']
        key = d, q
        base = dict(pool_scores[key]['frozen_scores'])
        model_scores = raw[key]['scores']
        relevant = gold[d][q]
        start = raw[key]['arms']['base']['ranking']
        old = row['accepted_order_scores']['full_top10']
        if base.keys() != model_scores.keys() or len(base) != 100:
            raise ValueError(f'Incomplete pool for {key}')
        item = {'domain': d, 'qid': q, 'active': row['active'],
                'accepted_order': row['accepted_orders']['full_top10'],
                'accepted_ndcg': old,
                'accepted_local_both_ndcg': row['accepted_order_scores']['top10_top10']}
        scale_ratio = max(pstdev(base[d] for d in item['accepted_order']), 1e-8) / max(
            pstdev(base.values()), 1e-8
        )
        item['accepted_frozen_std_ratio'] = None if row['active'] else scale_ratio
        if row['active']:
            scored = {}
            dense = list(base)
            if len(set(start)) != 10 or ndcg10(start, relevant) != row['six_pair']:
                raise ValueError(f'Original list disagrees with six-pair abstention for {key}')
            outsiders = [doc for doc in dense if doc not in start]
            for width in WIDTHS:
                keep = set(start) | set(outsiders[:width])
                docs = [doc for doc in dense if doc in keep]
                ranking, margin = apply(base, model_scores, docs)
                ndcg = ndcg10(ranking, relevant)
                archived = row['scores'].get(f'dense_{width}')
                if archived is not None and abs(ndcg - archived) > 1e-12:
                    raise ValueError(f'Replay differs for {key} at width {width}')
                scored[width] = (ranking, ndcg, margin)
            keep = set(start) | set(outsiders[:24])
            docs = [doc for doc in dense if doc in keep]
            norm = {f'{a}_{b}': ndcg10(apply(base, model_scores, docs, a, b)[0], relevant)
                    for a, b in NORMS}
            values = sorted(z(list(base.values())), reverse=True)
            preview = scored[12][0]
            item |= {'fallback_frozen_std_ratio': max(pstdev(base[d] for d in docs), 1e-8) /
                     max(pstdev(base.values()), 1e-8),
                     'rankings': {str(w): scored[w][0] for w in WIDTHS},
                     'ndcg': {str(w): scored[w][1] for w in WIDTHS},
                     'normalization': norm,
                     'signals': {
                         'upstream_overlap': len(set(start) & set(dense[:10])),
                         'upstream_margin': float(values[9] - values[10]),
                         'preview_overlap': len(set(start) & set(preview)),
                         'preview_margin': scored[12][2],
                     }}
        result.append(item)
    return result


def capacity(n, schedule):
    lower, middle, upper = schedule
    unit = middle - lower
    step = upper - middle
    if unit <= 0 or step % unit:
        raise ValueError('Expected a budget that balances with whole queries')
    ratio = step // unit
    high, residual = divmod(n, ratio + 1)
    return ratio * high, residual, high


def allocate(groups, signal, schedule):
    selected = {}
    for rows in groups.values():
        low, middle, high = capacity(len(rows), schedule)
        order = sorted(rows, key=lambda r: (r['signals'][signal], r['qid']))
        widths = [schedule[2]] * high + [24] * middle + [schedule[0]] * low
        assert sum(widths) == 24 * len(rows)
        selected.update({(r['domain'], r['qid']): w for r, w in zip(order, widths)})
    return selected


def oracle(rows, schedule):
    _, middle, high = capacity(len(rows), schedule)
    states = np.full((middle + 1, high + 1), -np.inf)
    states[0, 0] = 0.0
    for r in rows:
        a, b, c = (r['ndcg'][str(w)] for w in schedule)
        updated = states + a
        updated[1:, :] = np.maximum(updated[1:, :], states[:-1, :] + b)
        updated[:, 1:] = np.maximum(updated[:, 1:], states[:, :-1] + c)
        states = updated
    return float(states[-1, -1])


def evaluate(rows, schedule):
    groups = defaultdict(list)
    for row in rows:
        if row['active']:
            groups[row['domain']].append(row)
    for domain in {r['domain'] for r in rows}:
        groups.setdefault(domain, [])
    base = macro((r['domain'], r['ndcg']['24'] if r['active'] else r['accepted_ndcg']) for r in rows)
    active = sum(r['active'] for r in rows)
    arm_scores = {}
    for signal in SIGNALS:
        allocation = allocate(groups, signal, schedule)
        diffs = []
        for r in rows:
            w = allocation.get((r['domain'], r['qid']))
            if w:
                r.setdefault('assigned_widths', {}).setdefault(f'{schedule[0]}_{schedule[2]}', {})[signal] = w
            delta = r['ndcg'][str(w)] - r['ndcg']['24'] if w else 0.0
            diffs.append((r['domain'], delta))
        arm_scores[signal] = paired_delta(diffs)
    expected = []
    for r in rows:
        delta = 0.0
        if r['active']:
            count = len(groups[r['domain']])
            delta = sum(r['ndcg'][str(w)] * n / count
                        for w, n in zip(schedule, capacity(count, schedule))) - r['ndcg']['24']
        expected.append((r['domain'], delta))
    domain_totals = defaultdict(list)
    for r in rows:
        domain_totals[r['domain']].append(r)
    optimum = macro((d, (oracle(groups[d], schedule) + sum(r['accepted_ndcg'] for r in rs if not r['active'])) / len(rs))
                    for d, rs in domain_totals.items())
    return {'reference': base, 'vs_reference': arm_scores,
            'uniform_expected_vs_reference': paired_delta(expected),
            'allocation_oracle': optimum,
            'abstentions': active,
            'width_counts': {str(w): sum(capacity(len(items), schedule)[i] for items in groups.values())
                             for i, w in enumerate(schedule)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgeted', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--oof', type=Path, nargs='+', required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--oof-frozen', type=Path, required=True)
    parser.add_argument('--dev-frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    oof = build(args.budgeted, args.oof, args.oof_frozen, args.data_root, 'oof')
    dev = build(args.budgeted, [args.dev], args.dev_frozen, args.data_root, 'dev')
    report = {
        'protocol': 'Match the 34-document fallback budget separately in every domain. '
                    'Assign more candidates to low-margin or low-overlap queries. '
                    'Retain the same accepted-list rankings across all arms.',
        'schedules': {}, 'normalization_vs_fixed34': {}, 'normalization_by_route': {},
        'oof_per_query': oof, 'dev_per_query': dev,
    }
    for schedule in SCHEDULES:
        train, official = (evaluate(rows, schedule) for rows in (oof, dev))
        best = max(SIGNALS, key=lambda name: train['vs_reference'][name]['point'])
        report['schedules'][f'{schedule[0]}_{schedule[2]}'] = {
            'oof_selected_signal': best, 'oof': train, 'dev': official,
        }
    for split, rows in (('oof', oof), ('dev', dev)):
        report['normalization_vs_fixed34'][split] = {
            f'{a}_{b}': paired_delta([
                (r['domain'], r['normalization'][f'{a}_{b}'] - r['ndcg']['24'] if r['active'] else 0.0)
                for r in rows
            ]) for a, b in NORMS
        }
        changes = [
            (r['domain'],
             r['accepted_ndcg'] - r['accepted_local_both_ndcg'] if not r['active'] else 0.0,
             r['normalization']['full_local'] - r['normalization']['local_local'] if r['active'] else 0.0)
            for r in rows
        ]
        report['normalization_by_route'][split] = {
            'accepted_only': paired_delta([(d, a) for d, a, _ in changes]),
            'fallback_only': paired_delta([(d, b) for d, _, b in changes]),
            'accepted_minus_fallback': paired_delta([(d, a - b) for d, a, b in changes]),
            'median_frozen_local_to_full_std_ratio_accepted': median(
                r['accepted_frozen_std_ratio'] for r in rows if not r['active']
            ),
            'median_frozen_local_to_full_std_ratio_fallback': median(
                r['fallback_frozen_std_ratio'] for r in rows if r['active']
            ),
        }
    train_norm = report['normalization_vs_fixed34']['oof']
    report['oof_selected_deployable_normalization'] = max(
        ('local_local', 'full_local'), key=lambda name: train_norm[name]['point']
    )
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, separators=(',', ':'))
        stream.write('\n')
    for schedule, data in report['schedules'].items():
        print('schedule', schedule, 'OOF choice', data['oof_selected_signal'])
        for split in ('oof', 'dev'):
            scores = data[split]
            print(split, scores['reference'], scores['width_counts'], 'oracle', scores['allocation_oracle'])
            for key, val in scores['vs_reference'].items():
                print(key, val['point'], val['ci95'])
            print('uniform_expected', scores['uniform_expected_vs_reference']['point'])


if __name__ == '__main__':
    main()
