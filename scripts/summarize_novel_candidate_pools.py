"""Audit exact candidate replay and combine it with the accepted-list result."""
import argparse
import json
from pathlib import Path

from audit_anchor_dev import coverage, macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels, read_steps
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


def fit_proxy(rows, arms):
    folds = sorted({r['fold'] for r in rows})
    selected = {}
    for held in folds:
        selected[held] = max(arms, key=lambda a: macro(
            (r['domain'], r['scores'][a]) for r in rows if r['fold'] != held
        ))
    gain = paired_delta([(r['domain'], r['scores'][selected[r['fold']]] - r['scores']['dense34']) for r in rows])
    return {'choice_by_target_fold': selected, 'change': gain}


def evaluate(rows, prior, protected, sizes, labels, steps=None):
    active = index({'per_query': rows})
    accepted = index(protected)
    before = index(prior)
    if active.keys() != before.keys() or not accepted.keys().isdisjoint(active):
        raise ValueError('Query or branch membership differs from the baseline')
    full_keys = accepted.keys() | active.keys()
    effects = {}
    for arm, pool in sizes.items():
        scores, diffs, recalls = [], [], []
        step_scores = []
        for domain, qid in sorted(full_keys):
            key = domain, qid
            gold = labels[domain][qid]
            if key in active:
                row = active[key]
                old = before[key]['rankings']['max_1']
                if row['rankings']['dense34'] != old:
                    raise ValueError(f'Baseline ranking differs for {key}')
                ranking = row['rankings'][arm]
                change = row['oracle'][arm] - row['oracle']['dense34']
                recalls.append((domain, change))
                raw = row['scores'][arm]
            else:
                old = ranking = accepted[key]['rankings']['1']
                raw = accepted[key]['scores']['1']
                recalls.append((domain, 0.))
            value = ndcg10(ranking, gold)
            if len(ranking) != 10 or len(set(ranking)) != 10 or abs(raw-value) > 1e-12:
                raise ValueError(f'Ranking or score differs for {key}/{arm}')
            scores.append((domain, value))
            diffs.append((domain, value-ndcg10(old, gold)))
            if steps and steps[domain].get(qid):
                step_scores.append((domain, coverage(ranking, steps[domain][qid])))
        effects[arm] = {
            'ndcg': macro(scores), 'change': paired_delta(diffs),
            'candidate_oracle_change': paired_delta(recalls),
            'coverage': macro(step_scores) if step_scores else None,
            'extra_prompts_per_query': 2 * (pool - 34) * len(active) / len(full_keys),
            'changed_candidates_on_abstentions': sum(r['novel_members_changed'].get(arm, 0.) for r in rows) / len(rows),
        }
    return effects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'prior', 'protected', 'proxy', 'data-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    replay = load(args.replay)
    prior = load(args.prior)
    protected = load(args.protected)
    sizes = replay['pool_sizes']
    proxy = load(args.proxy)
    proxy_rows = [r for r in replay['per_query'] if r['sample'] == 'proxy']
    keys = lambda rows: {(r['domain'], r['qid']) for r in rows}
    if keys(proxy_rows) != keys(proxy['per_query']):
        raise ValueError('The new proxy uses different queries')
    proxy_index = index(proxy)
    if any(abs(r['scores']['dense34'] - proxy_index[r['domain'], r['qid']]['scores']['max_1']) > 1e-12
           for r in proxy_rows):
        raise ValueError('The same proxy baseline was not reproduced')
    equal = [a for a, width in sizes.items() if width == 34]
    report = {key: replay[key] for key in ('selected_equal_budget', 'selected_any_budget')}
    report['crossfit'] = {'equal': fit_proxy(proxy_rows, equal), 'any': fit_proxy(proxy_rows, sizes)}
    report['proxy_conditional'] = replay['scores_conditioned_on_abstention']['proxy']
    for split, label in (('oof', 'train'), ('dev', 'dev')):
        rows = [r for r in replay['per_query'] if r['sample'] == split]
        domains = {r['domain'] for r in rows + protected[split]['per_query']}
        gold = {d: read_qrels(args.data_root / d / f'qrels_{label}.txt') for d in domains}
        steps = {d: read_steps(args.data_root / d / 'qrels_steps_dev.txt') for d in domains} if split == 'dev' else None
        report[split] = evaluate(rows, prior[split], protected[split], sizes, gold, steps)
        for name, values in report[split].items():
            print(split, name, values['ndcg'], values['change']['ci95'],
                  values['candidate_oracle_change']['point'], values['extra_prompts_per_query'], flush=True)
    print('Selected:', report['selected_equal_budget'], report['selected_any_budget'])
    print('Cross-fit:', {k: (v['choice_by_target_fold'], v['change']['point'], v['change']['ci95'])
                         for k, v in report['crossfit'].items()})
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
