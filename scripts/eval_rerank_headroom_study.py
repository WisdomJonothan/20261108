"""Select and evaluate a matched evidence-reranking policy on fixed pools."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import coverage, macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels, read_steps
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import discounted_changes, index


POLICIES = {
    'reference': (10, 0., False),
    'reorder': (10, .5, False),
    'reorder_head_fixed': (10, .5, True),
    'near_conservative': (20, .3, False),
    'near_balanced': (20, .5, False),
    'near_judge': (20, 1., False),
    'deeper_conservative': (30, .3, False),
    'deeper_balanced': (30, .5, False),
    'deeper_judge': (30, 1., False),
    'deeper_head_fixed': (30, .5, True),
}


def rank(docs, score, policy):
    width, weight, protect_head = POLICIES[policy]
    if policy == 'reference':
        return docs[:10]
    # Both sources use their standard deviation in the same pool.
    prior = z(1 / np.log2(2 + np.arange(width)))
    judge = z([score[d] for d in docs[:width]])
    blend = (1 - weight) * prior + weight * judge
    order = np.argsort(-blend, kind='stable')
    if protect_head:
        order = [0, *[int(i) for i in order if i != 0]]
    return [docs[i] for i in order[:10]]


def registers(paths):
    reports = {}
    for path in paths:
        raw = load(path)
        mode = raw['evidence_mode']
        # Results may be partitioned by domain; an overlap is never a valid merge.
        if mode not in reports:
            reports[mode] = {'scores': {}, 'inference': []}
        scores = index(raw)
        if not reports[mode]['scores'].keys().isdisjoint(scores):
            raise ValueError(f'Duplicate query scores for evidence mode {mode}')
        reports[mode]['scores'].update(scores)
        reports[mode]['inference'].append(raw['inference'])
    return reports


def evaluate(manifest, reports, data):
    gold = {(d, s): read_qrels(data / d / f'qrels_{s}.txt')
            for d, s in {(r['domain'], r['split']) for r in manifest.values()}}
    for name, report in reports.items():
        if report['scores'].keys() != manifest.keys():
            raise ValueError(f'Missing or unexpected queries for {name}')
    results = []
    for (domain, qid), row in manifest.items():
        docids = row['docs']
        labels = gold[domain, row['split']][qid]
        if len(set(docids)) != len(docids) or len(docids) != 30:
            raise ValueError(f'Unexpected evaluation shortlist for {domain}/{qid}')
        modes = {mode: report['scores'][domain, qid]['scores'] for mode, report in reports.items()}
        if any(set(docids) != item.keys() for item in modes.values()):
            raise ValueError(f'Unexpected score membership for {domain}/{qid}')
        order = {'baseline': docids[:10]}
        for mode, scores in modes.items():
            for arm in POLICIES:
                if arm != 'reference':
                    order[f'{mode}/{arm}'] = rank(docids, scores, arm)
        values = {arm: ndcg10(ranking, labels) for arm, ranking in order.items()}
        results.append({k: row[k] for k in ('domain', 'qid', 'split', 'cohort', 'actual_reference')} |
                       {'rankings': order, 'scores': values})
    return results


def choose(rows, arms):
    return max(arms, key=lambda key: macro((r['domain'], r['scores'][key]) for r in rows))


def stats(rows, arms):
    gains = [(r['domain'], r['scores'][arm] - r['scores']['baseline'])
             for r, arm in zip(rows, arms, strict=True)]
    return {'score': macro((r['domain'], r['scores'][arm]) for r, arm in zip(rows, arms, strict=True)),
            'change': paired_delta(gains),
            'ranks_changed': sum(r['rankings'][arm] != r['rankings']['baseline']
                                 for r, arm in zip(rows, arms, strict=True))}


def summary(rows, chosen):
    out = {arm: stats(rows, [arm] * len(rows)) for arm in rows[0]['scores']}
    out['selected'] = stats(rows, [chosen[r['cohort']] for r in rows])
    return out


def select(training, modes):
    arms = tuple(training[0]['scores'])
    per_fold = {fold: choose([r for r in training if r['cohort'] != fold], arms)
                for fold in sorted({r['cohort'] for r in training})}
    best_by_mode = {m: choose(training, [a for a in arms if a.startswith(m + '/')]) for m in modes}
    exact = [r for r in training if r['actual_reference']]
    selected = choose(training, arms)
    return {
        'selected': selected, 'choices_by_held_fold': per_fold, 'best_by_mode': best_by_mode,
        'training_summary': summary(training, per_fold),
        'actual_reference_training_summary': summary(exact, per_fold),
        'pooled_selected_on_actual_reference': stats(exact, [selected] * len(exact)),
    }


def diagnose(rows, selected, data, reference='baseline'):
    domains = {row['domain'] for row in rows}
    gold = {d: read_qrels(data / d / 'qrels_dev.txt') for d in domains}
    steps = {d: read_steps(data / d / 'qrels_steps_dev.txt') for d in domains}
    counts = Counter()
    parts = {k: [] for k in ('added_documents', 'removed_documents', 'retained_documents')}
    step_scores = {k: [] for k in (reference, 'selected')}
    for row in rows:
        d, q = row['domain'], row['qid']
        previous, new = (row['rankings'][k] for k in (reference, selected))
        relevant = gold[d][q]
        before, after = set(previous) & relevant, set(new) & relevant
        counts['new_positives'] += len(after - before)
        counts['lost_positives'] += len(before - after)
        counts['zeroes_recovered'] += not before and bool(after)
        counts['new_zeroes'] += bool(before) and not after
        counts['rank1_changed'] += previous[0] != new[0]
        effects = discounted_changes(previous, new, relevant)
        for name, values in parts.items():
            values.append((d, effects[name]))
        annotated = steps[d].get(q)
        if annotated:
            step_scores[reference].append((d, coverage(previous, annotated)))
            step_scores['selected'].append((d, coverage(new, annotated)))
    return {'counts': dict(counts), 'ndcg_contributions': {k: macro(v) for k, v in parts.items()},
            'step_coverage': {k: macro(v) for k, v in step_scores.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-root', 'selection', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=(name != 'selection'))
    parser.add_argument('--scores', nargs='+', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    reports = registers(args.scores)
    raw = load(args.manifest)['queries']
    manifest = index({'per_query': raw})
    rows = evaluate(manifest, reports, args.data_root)
    population = {r['split'] for r in rows}
    if args.selection:
        if population != {'dev'}:
            raise ValueError('Apply a locked training policy to official dev only')
        selection = load(args.selection)
        selected = selection['selected']
        if selected not in rows[0]['scores']:
            raise ValueError('The locked policy needs a missing judge')
        result = {'selected': selected, 'training_input': args.selection.name,
                  'dev': summary(rows, {'dev': selected}),
                  'selected_diagnostics': diagnose(rows, selected, args.data_root)}
        if len(rows) != 519 or abs(result['dev']['baseline']['score'] - .39269165354596675) > 1e-12:
            raise ValueError('Official dev baseline changed')
    else:
        if population != {'train'}:
            raise ValueError('Policy selection requires training-fold queries')
        result = select(rows, reports)
        print('Best train policies by mode', result['best_by_mode'], flush=True)
    result.update({'protocol': POLICIES, 'per_query': rows, 'inference': {
        name: report['inference'] for name, report in reports.items()
    }})
    print('Selected', result['selected'], flush=True)
    for label in ('training_summary', 'actual_reference_training_summary', 'dev'):
        if label in result:
            print(label, {k: round(v['score'], 6) for k, v in result[label].items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
