"""Locate accepted-list weight gains by position and verify matched 4B/8B transfer."""
import argparse
import json
from pathlib import Path

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import discounted_changes, index
from replay_anchor_fusion_calibration import build, rank, score, weight

PROTECTED = (0, 1, 3, 10)
BANDS = {'rank1': (0, 1), 'ranks2_3': (1, 3), 'ranks4_6': (3, 6), 'ranks7_10': (6, 10)}


def orders(row):
    before, after = (rank(row, weight(alpha)) for alpha in (.3, .6))
    if set(before) != set(after):
        raise ValueError('Expected the same ten documents in both arms')
    return before, after


def protect(before, after, number):
    head = before[:number]
    return head + [doc for doc in after if doc not in head]


def scope(rows, baseline):
    result = []
    for row in rows:
        if row['route'] != 'accepted':
            continue
        before, after = orders(row)
        bands = discounted_changes(before, after, row['gold'])['rank_delta']
        scores = {}
        ranks = {}
        for number in PROTECTED:
            ranking = protect(before, after, number)
            scores[str(number)] = score(row, ranking)
            ranks[str(number)] = ranking
        if abs(sum(bands) - (scores['0'] - scores['10'])) > 1e-12:
            raise ValueError(f'Rank effects do not add up for {row["qid"]}')
        result.append({
            'domain': row['domain'], 'qid': row['qid'], 'rank1_changed': before[0] != after[0],
            'scores': scores, 'rankings': ranks, 'rank_effects': bands,
        })
    metrics = {
        str(n): {'ndcg': macro((r['domain'], r['scores'][str(n)]) for r in result),
                 'change': paired_delta([(r['domain'], r['scores'][str(n)] - r['scores']['10'])
                                          for r in result])}
        for n in PROTECTED
    }
    per_query = {(r['domain'], r['qid']): r for r in result}
    full = {}
    for number in PROTECTED:
        effect = []
        for row in rows:
            r = per_query.get((row['domain'], row['qid']))
            delta = 0. if r is None else r['scores'][str(number)] - r['scores']['10']
            effect.append((row['domain'], delta))
        paired = paired_delta(effect)
        full[str(number)] = {'ndcg': baseline + paired['point'], 'change': paired}
    attribution = {
        name: macro((r['domain'], sum(r['rank_effects'][a:b])) for r in result)
        for name, (a, b) in BANDS.items()
    }
    return {'queries': len(result), 'domains': len({r['domain'] for r in result}),
            'protected_first_positions': metrics, 'full_policy': full, 'contribution_by_rank': attribution,
            'queries_with_new_rank1': sum(r['rank1_changed'] for r in result), 'per_query': result}


def crossfit_guard(train, audit, baseline):
    accepted = index(audit)
    selected = {}
    differences = []
    for cohort in (0, 1):
        training = [accepted[r['domain'], r['qid']] for r in train
                    if r['cohort'] != cohort and r['route'] == 'accepted']
        n = max(PROTECTED, key=lambda k: (
            macro((r['domain'], r['scores'][str(k)]) for r in training), k
        ))
        selected[str(cohort)] = n
        for r in train:
            if r['cohort'] != cohort:
                continue
            held = accepted.get((r['domain'], r['qid']))
            delta = 0. if held is None else held['scores'][str(n)] - held['scores']['10']
            differences.append((r['domain'], delta))
    paired = paired_delta(differences)
    return {'chosen_protection_by_target_cohort': selected,
            'ndcg': baseline + paired['point'], 'change': paired}


def compare(dev, fresh, previous):
    frozen = {(r['domain'], r['qid']): r for r in dev if r['route'] == 'accepted'}
    small = index(fresh)
    old_subset = {(r['domain'], r['qid']) for r in previous['dev']['per_query']}
    if frozen.keys() != small.keys() or not old_subset <= small.keys():
        raise ValueError('Expected score data for every accepted dev query')
    rows = []
    for key, row in frozen.items():
        other = small[key]
        thirty, sixty = orders(row)
        if set(thirty) != set(other['accepted']):
            raise ValueError(f'Candidate membership differs for {key}')
        guard4 = protect(other['rankings']['0.3'], other['rankings']['0.6'], 1)
        scores = {'4b': {a: other['ndcg'][a] for a in ('0.3', '0.6')} | {'protect1': score(row, guard4)},
                  '8b': {'0.3': score(row, thirty), '0.6': score(row, sixty),
                         'protect1': score(row, protect(thirty, sixty, 1))}}
        rows.append({'domain': key[0], 'qid': key[1], 'in_old_matched_subset': key in old_subset,
                     'scores': scores})

    def summary(items):
        ml = {}
        for model in ('4b', '8b'):
            ml[model] = {
                'ndcg': {a: macro((r['domain'], r['scores'][model][a]) for r in items)
                         for a in ('0.3', '0.6', 'protect1')},
                'change': paired_delta([(r['domain'], r['scores'][model]['0.6'] - r['scores'][model]['0.3'])
                                        for r in items]),
                'guard_change': paired_delta([
                    (r['domain'], r['scores'][model]['protect1'] - r['scores'][model]['0.3']) for r in items
                ]),
            }
        return {'queries': len(items), 'domains': len({r['domain'] for r in items}),
                'models': ml,
                'weight_change_8b_minus_4b': paired_delta([
                    (r['domain'], (r['scores']['8b']['0.6'] - r['scores']['8b']['0.3']) -
                     (r['scores']['4b']['0.6'] - r['scores']['4b']['0.3'])) for r in items
                ])}

    return {'all': summary(rows),
            'old_subset_fresh_4b': summary([r for r in rows if r['in_old_matched_subset']]),
            'newly_covered': summary([r for r in rows if not r['in_old_matched_subset']]),
            'per_query': rows}


def fresh_train_guard(fresh, data):
    rows = fresh['per_query']
    labels = {d: read_qrels(data / d / 'qrels_train.txt') for d in {r['domain'] for r in rows}}
    effect = []
    for r in rows:
        d, q = r['domain'], r['qid']
        guard = protect(r['rankings']['0.3'], r['rankings']['0.6'], 1)
        effect.append((d, ndcg10(guard, labels[d][q]) - r['ndcg']['0.3']))
    paired = paired_delta(effect)
    return {'ndcg': fresh['summary']['macro']['0.3'] + paired['point'], 'change': paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgeted', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--oof', type=Path, nargs=2, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--oof-frozen', type=Path, required=True)
    parser.add_argument('--dev-frozen', type=Path, required=True)
    parser.add_argument('--fresh-4b', type=Path)
    parser.add_argument('--fresh-train-4b', type=Path)
    parser.add_argument('--previous-match', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.fresh_4b is None) != (args.previous_match is None):
        parser.error('The full comparison also needs the prior matched subset.')
    archive = load(args.budgeted)
    train = build(archive, args.oof, args.oof_frozen, args.data_root, 'oof')
    dev = build(archive, [args.dev], args.dev_frozen, args.data_root, 'dev')
    baseline = archive['accepted_reorder_on_top_of_selected_24']
    result = {'oof': scope(train, baseline['oof']['full_top10']['ndcg']),
              'dev': scope(dev, baseline['dev']['full_top10']['ndcg'])}
    result['oof_selected_protected_positions'] = max(
        PROTECTED, key=lambda n: (result['oof']['protected_first_positions'][str(n)]['ndcg'], n)
    )
    result['crossfit_oof_guard'] = crossfit_guard(train, result['oof'], baseline['oof']['full_top10']['ndcg'])
    if args.fresh_4b:
        result['matched_full_dev'] = compare(dev, load(args.fresh_4b), load(args.previous_match))
    if args.fresh_train_4b:
        result['fresh_training_4b_guard'] = fresh_train_guard(load(args.fresh_train_4b), args.data_root)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')
    for split in ('oof', 'dev'):
        rows = result[split]
        print(split, rows['queries'], rows['queries_with_new_rank1'],
              {k: a['ndcg'] for k, a in rows['protected_first_positions'].items()},
              rows['contribution_by_rank'])
        print('full', {k: (a['ndcg'], a['change']['ci95']) for k, a in rows['full_policy'].items()})
    print('OOF-selected number of protected positions:', result['oof_selected_protected_positions'])
    print('Cross-fit guard:', result['crossfit_oof_guard'])
    for name, sample in result.get('matched_full_dev', {}).items():
        if name != 'per_query':
            print(name, 'queries', sample['queries'])
            for key, item in sample['models'].items():
                print(key, item['ndcg'], item['change']['point'], item['change']['ci95'], 'guard', item['guard_change']['ci95'])
            contrast = sample['weight_change_8b_minus_4b']
            print('contrast', contrast['point'], contrast['ci95'], contrast['domain_bootstrap_ci95'])


if __name__ == '__main__':
    main()
