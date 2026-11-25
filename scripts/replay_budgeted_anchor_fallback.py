"""Replay smaller 8B pools with candidate and normalization budgets fixed."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import ordered_scores, paired_delta, z
from replay_anchor_fallback import index

WIDTHS = (6, 12, 24, 48)
SOURCES = ('dense', 'fusion', 'rrf')
SCOPES = (('top10', 'top10'), ('full', 'top10'), ('top10', 'full'), ('full', 'full'))


def shortlist_sources(base, start):
    dense = [doc for doc, _ in base]
    fused = [doc for doc, _ in sorted(base, key=lambda item: -item[1])]
    ranks = [{doc: rank for rank, doc in enumerate(order, 1)} for order in (dense, fused)]
    rrf = sorted(dense, key=lambda doc: -sum(1 / (60 + rank[doc]) for rank in ranks))
    return {name: [doc for doc in ranking if doc not in start]
            for name, ranking in zip(SOURCES, (dense, fused, rrf))}


def accepted_orders(base, anchor, start):
    docs = [doc for doc, _ in base]
    keep = [i for i, doc in enumerate(docs) if doc in start]
    frozen = np.array([score for _, score in base])
    judge = np.array([anchor[doc] for doc in docs])
    frozen_z = {'top10': z(frozen[keep]), 'full': z(frozen)[keep]}
    judge_z = {'top10': z(judge[keep]), 'full': z(judge)[keep]}
    orders = {}
    for a, b in SCOPES:
        combined = .7 * frozen_z[a] + .3 * judge_z[b]
        orders[f'{a}_{b}'] = [docs[keep[i]] for i in np.argsort(-combined, kind='stable')]
    return orders


def replay(paths, scope, frozen, data, split):
    records = {}
    for path in paths:
        raw = load(path)
        if raw['split'] != split or raw['pool_depth'] != 100:
            raise ValueError(f'Expected a depth-100 {split} cache in {path}')
        cohort = index(raw)
        if not cohort.keys().isdisjoint(records):
            raise ValueError('Input cohorts overlap')
        records.update(cohort)
    abstentions = index(load(scope)['starts']['abstention'])
    bases = index(load(frozen))
    if records.keys() != abstentions.keys() or not records.keys() <= bases.keys():
        raise ValueError('Query IDs differ across score and ranking inputs')
    gold = {d: read_qrels(data / d / f'qrels_{split}.txt') for d, _ in records}
    output = []
    for key, row in records.items():
        domain, qid = key
        relevant = gold[domain][qid]
        base = bases[key]['frozen_scores']
        anchor = row['scores']
        archived = abstentions[key]
        start = archived['rankings']['start']
        if len(base) != 100 or {doc for doc, _ in base} != anchor.keys() or (
            ordered_scores(base, anchor)[:10] != row['arms']['base_plus_anchor30']['ranking']
        ):
            raise ValueError(f'Cannot reconstruct original depth-100 fusion for {key}')
        baseline = ndcg10(start, relevant)
        full = ndcg10(archived['rankings']['unrestricted'], relevant)
        if max(abs(baseline - archived['scores']['start']),
               abs(full - archived['scores']['unrestricted'])) > 1e-12:
            raise ValueError(f'Cached nDCG differs from binary labels for {key}')
        choices = shortlist_sources(base, set(start))
        scores, ceilings, policies = {}, {}, {}
        for name, outsiders in choices.items():
            for width in WIDTHS:
                arm = f'{name}_{width}'
                pool = set(start) | set(outsiders[:width])
                ranking = start
                if archived['active']:
                    if len(pool) != 10 + width or row['anchor'] not in pool:
                        raise ValueError(f'Shortlist size or anchor is invalid for {key}')
                    ranking = ordered_scores([(d, v) for d, v in base if d in pool], anchor)[:10]
                scores[arm] = ndcg10(ranking, relevant)
                ceilings[arm] = ndcg10(sorted(pool & relevant), relevant) if archived['active'] else baseline
                policies[arm] = ranking
        accepted = ({f'{a}_{b}': start for a, b in SCOPES} if archived['active']
                    else accepted_orders(base, anchor, start))
        output.append({'domain': domain, 'qid': qid, 'active': archived['active'],
                       'six_pair': baseline, 'full': full, 'scores': scores,
                       'oracle': ceilings, 'rankings': policies,
                       'accepted_orders': accepted,
                       'accepted_order_scores': {name: ndcg10(ranking, relevant)
                                                 for name, ranking in accepted.items()},
                       'accepted_prompts': 0 if archived['active'] else 2 * (10 - int(row['anchor'] in start))})
    return output


def summarize(rows):
    reference = {k: macro((r['domain'], r[k]) for r in rows) for k in ('six_pair', 'full')}
    active = sum(r['active'] for r in rows)
    arms = {}
    for name in rows[0]['scores']:
        width = int(name.rsplit('_', 1)[1])
        arms[name] = {
            'ndcg': macro((r['domain'], r['scores'][name]) for r in rows),
            'oracle': macro((r['domain'], r['oracle'][name]) for r in rows),
            'versus_six_pair': paired_delta([(r['domain'], r['scores'][name] - r['six_pair']) for r in rows]),
            'versus_full': paired_delta([(r['domain'], r['scores'][name] - r['full']) for r in rows]),
            'mean_prompts': 12 + 2 * (9 + width) * active / len(rows),
            'recovered_zeroes': sum(r['six_pair'] == 0 < r['scores'][name] for r in rows),
            'new_zeroes': sum(r['scores'][name] == 0 < r['six_pair'] for r in rows),
        }
    return {'queries': len(rows), 'abstained': active, 'references': reference,
            'full_mean_prompts': 12 + 198 * active / len(rows), 'arms': arms}


def accepted_reorder(rows, arm, scope):
    def score(row):
        return row['scores'][arm] if row['active'] else row['accepted_order_scores'][scope]

    return {
        'ndcg': macro((r['domain'], score(r)) for r in rows),
        'paired_vs_shortlist_only': paired_delta([
            (r['domain'], score(r) - r['scores'][arm]) for r in rows
        ]),
        'paired_vs_six_pair': paired_delta([
            (r['domain'], score(r) - r['six_pair']) for r in rows
        ]),
        'extra_prompts_per_query': (
            198 * sum(not r['active'] for r in rows) / len(rows) if scope.endswith('_full')
            else sum(r['accepted_prompts'] for r in rows) / len(rows)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--oof', type=Path, nargs='+', required=True)
    parser.add_argument('--dev', type=Path, required=True)
    for name in ('oof-frozen', 'dev-frozen', 'oof-scope', 'dev-scope'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train = replay(args.oof, args.oof_scope, args.oof_frozen, args.data_root, 'train')
    dev = replay([args.dev], args.dev_scope, args.dev_frozen, args.data_root, 'dev')
    oof, official = summarize(train), summarize(dev)
    chosen = {
        str(width): max((f'{source}_{width}' for source in SOURCES), key=lambda arm: oof['arms'][arm]['ndcg'])
        for width in WIDTHS
    }
    chosen_names = set(chosen.values())
    for rows in (train, dev):
        for row in rows:
            row['rankings'] = {key: value for key, value in row['rankings'].items() if key in chosen_names}
    report = {
        'protocol': 'Keep the six-pair result on accepted queries. For abstentions protect its ten '
                    'documents and add a fixed number of outsiders selected by dense order, frozen '
                    'fusion, or RRF. Recompute normalization inside each pool and apply 70:30. '
                    'Select one source per width on training holdouts. Prompt counts are replay budgets.',
        'oof_selected_per_width': chosen, 'oof': oof, 'dev': official,
        'accepted_reorder_on_top_of_selected_24': {
            'arm': chosen['24'],
            'oof': {f'{a}_{b}': accepted_reorder(train, chosen['24'], f'{a}_{b}') for a, b in SCOPES},
            'dev': {f'{a}_{b}': accepted_reorder(dev, chosen['24'], f'{a}_{b}') for a, b in SCOPES},
        },
        'oof_per_query': train, 'dev_per_query': dev,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, separators=(',', ':'))
        stream.write('\n')
    for name, item in official['arms'].items():
        print(name, 'oof', round(oof['arms'][name]['ndcg'], 6), 'dev', round(item['ndcg'], 6),
              'oracle', round(item['oracle'], 6), 'prompts', round(item['mean_prompts'], 2),
              'paired vs 100', item['versus_full']['ci95'])
    print('Selected on the training holdouts:', chosen)
    print('Official references:', official['references'])


if __name__ == '__main__':
    main()
