"""Select a Qwen evidence scorer and ranking policy on held-out training queries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_rerank_headroom_study import diagnose, rank, registers
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index


# Candidate count, adapted fraction of judge score, total judge weight, protect rank one.
POLICIES = {
    'direct30': (30, 1., 1., False),
    'adaptive20_protected': (20, 1., .50, True),
    'adaptive30_protected': (30, 1., .50, True),
    'adaptive30_protected_strong': (30, 1., .70, True),
    'adaptive30_free': (30, 1., .50, False),
    'adaptive30_free_strong': (30, 1., .70, False),
    'matched_ensemble30': (30, .50, .50, True),
    'matched_ensemble30_strong': (30, .50, .70, True),
    'adaptive45_protected': (45, 1., .65, True),
    'adaptive60_protected': (60, 1., .65, True),
    'adaptive60_unrestricted': (60, 1., .85, False),
}
CURRENT_LIST_POLICIES = {
    'start_from_repair_m25': (30, 1., .25, True),
    'start_from_repair_m40': (30, 1., .40, True),
    'start_from_repair_m55': (30, 1., .55, False),
}


def candidate_judges(paths):
    cohorts = defaultdict(dict)
    metadata = {}
    inference = defaultdict(list)
    for path in paths:
        result = load(path)
        evidence_mode = result['evidence_mode']
        if evidence_mode not in ('dispersed', 'expanded'):
            raise ValueError('Unexpected evidence inputs for the model comparison')
        if 'adapter' in result:
            if evidence_mode != 'dispersed':
                raise ValueError('The adapter comparison needs the fixed two-excerpt inputs')
            identity = result['adapter']
            tag = f"epoch{identity['training']['epoch']}"
        else:
            identity = {'model': result['model'], 'revision': result['model_revision']}
            if evidence_mode == 'expanded':
                identity['evidence_mode'] = evidence_mode
            tag = 'pretrained' if evidence_mode == 'dispersed' else 'context_expanded'
            if 'focus' in result:
                identity['focus'] = result['focus']
                tag = 'focus_' + result['focus']
            if 'score_policy' in result:
                identity['score_policy'] = result['score_policy']
                tag = result['score_policy']
        if tag in metadata and metadata[tag] != identity:
            raise ValueError(f'Model identity differs among shards for {tag}')
        metadata[tag] = identity
        per_query = index(result)
        for key, row in per_query.items():
            if key not in cohorts[tag]:
                cohorts[tag][key] = row
                continue
            earlier = cohorts[tag][key]
            if (not earlier['scores'].keys().isdisjoint(row['scores'])
                    or (earlier['split'], earlier['cohort']) != (row['split'], row['cohort'])):
                raise ValueError(f'Overlapping document scores or split mismatch for {tag}/{key}')
            earlier['scores'].update(row['scores'])
        inference[tag].append(result['inference'])
    return cohorts, metadata, inference


def blend(docs, base, adapted, policy, *, top_k=10):
    width, trained, alpha, protect = policy
    shortlist = docs[:width]
    prior = z(1 / np.log2(2 + np.arange(width)))
    new = z([adapted[d] for d in shortlist])
    previous = z([base[d] for d in shortlist]) if trained < 1. else 0.
    combined = (1-alpha)*prior + alpha*((1-trained)*previous + trained*new)
    indices = np.argsort(-combined, kind='stable')
    if protect:
        indices = [0, *[int(i) for i in indices if i != 0]]
    return [shortlist[i] for i in indices[:top_k]]


def prepare(manifest, references, controls, fitted, split):
    queries = index({'per_query': load(manifest)['queries']})
    current = index({'per_query': references[f'{"oof" if split == "train" else "dev"}_per_query']})
    if controls.keys() != queries.keys() or any(q.keys() != queries.keys() for q in fitted.values()):
        raise ValueError('The matched score files must cover the same manifest')
    if current.keys() != queries.keys():
        raise ValueError('Candidate manifest differs from the previous scored repair')
    results = []
    for key, row in queries.items():
        docs = row['docs']
        ref = current[key]
        selected = (references['choices'][row['cohort']] if split == 'train' else references['selected'])
        if (ref['rankings']['baseline'] != docs[:10] or len(set(docs)) != len(docs)
                or len(docs) not in (30, 60)):
            raise ValueError(f'Unexpected fixed shortlist for {key}')
        scores = controls[key]['scores']
        orders = {'current': ref['rankings'][selected], 'baseline': docs[:10],
                  'unadapted_direct': sorted(docs[:30], key=lambda d: -scores[d])[:10],
                  'unadapted_matched': rank(docs[:30], scores, 'deeper_head_fixed')}
        if orders['unadapted_matched'] != ref['rankings']['keep_dispersed']:
            raise ValueError(f'Cached unadapted comparator differs for {key}')
        for epoch, cache in fitted.items():
            new = cache[key]['scores']
            if scores.keys() != set(docs[:30]) or new.keys() != set(docs):
                raise ValueError(f'Candidate scores differ for {key}')
            for name, policy in POLICIES.items():
                if policy[0] <= len(docs):
                    orders[f'{epoch}/{name}'] = blend(docs, scores, new, policy)
            old = orders['current']
            values = .7*z(1/np.log2(2+np.arange(10))) + .3*z([new[d] for d in old])
            orders[f'{epoch}/same_ten_reorder'] = [old[i] for i in np.argsort(-values, kind='stable')]
            if epoch in ('pretrained', 'context_expanded', 'full_plus_extractive_max'):
                anchored = old + [d for d in docs if d not in old]
                for name, policy in CURRENT_LIST_POLICIES.items():
                    orders[f'{epoch}/{name}'] = blend(anchored, scores, new, policy)
        results.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                       {'rankings': orders})
    return results


def grade(rows, split, data):
    if {r['split'] for r in rows} != {split}:
        raise ValueError(f'Expected {split} rows')
    gold = {d: read_qrels(data / d / f'qrels_{split}.txt') for d in {r['domain'] for r in rows}}
    return [r | {'scores': {arm: ndcg10(ranking, gold[r['domain']][r['qid']])
                            for arm, ranking in r['rankings'].items()}} for r in rows]


def metric(rows, choices):
    return macro((r['domain'], r['scores'][choice]) for r, choice in zip(rows, choices, strict=True))


def choose(rows, arms):
    return max(arms, key=lambda arm: metric(rows, [arm]*len(rows)))


def summarize(rows, selected, by_fold=None):
    choices = {arm: [arm]*len(rows) for arm in rows[0]['scores']}
    choices['selected'] = [by_fold[r['cohort']] if by_fold else selected for r in rows]
    return {
        arm: {'score': metric(rows, decision), 'change': paired_delta([
            (r['domain'], r['scores'][c] - r['scores']['current']) for r, c in zip(rows, decision, strict=True)
        ])}
        for arm, decision in choices.items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'repair', 'selection', 'data-root', 'output'):
        ap.add_argument(f'--{name}', type=Path, required=name != 'selection')
    ap.add_argument('--control-scores', type=Path, nargs='+', required=True)
    ap.add_argument('--adapter-scores', '--candidate-scores', type=Path, nargs='+', required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a fresh output path')
    fitted, metadata, inference = candidate_judges(args.adapter_scores)
    reference = load(args.repair)
    base = registers(args.control_scores)
    if set(base) != {'dispersed'}:
        raise ValueError('Expected a two-excerpt zero-shot reference')
    if args.selection:
        trained = load(args.selection)
        selected = trained['selected']
        needed = selected.split('/')[0]
        if needed not in metadata or metadata[needed] != trained['adapters'][needed]:
            raise ValueError('Official-dev checkpoint differs from the training choice')
        rows = grade(prepare(args.manifest, reference, base['dispersed']['scores'], fitted, 'dev'),
                     'dev', args.data_root)
        result = {'selected': selected, 'dev': summarize(rows, selected),
                  'diagnostics': diagnose(rows, selected, args.data_root, reference='current')}
        if len(rows) != 519 or abs(result['dev']['current']['score'] - .3958942538410169) > 1e-12:
            raise ValueError('Full official research baseline changed')
        print('Dev', {arm: v['score'] for arm, v in result['dev'].items()}, flush=True)
    else:
        rows = grade(prepare(args.manifest, reference, base['dispersed']['scores'], fitted, 'train'),
                     'train', args.data_root)
        if len(rows) != 186:
            raise ValueError('Unexpected held-out sample')
        arms = ['current'] + [arm for arm in rows[0]['scores'] if arm.split('/')[0] in fitted]
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], arms)
                   for f in sorted({r['cohort'] for r in rows})}
        selected = choose(rows, arms)
        matched = [r for r in rows if r['actual_reference']]
        result = {'selected': selected, 'by_fold': by_fold, 'training': summarize(rows, selected, by_fold),
                  'original_65': summarize(matched, selected, by_fold)}
        print('Selected', selected, by_fold, flush=True)
        print('Training', {arm: v['score'] for arm, v in result['training'].items()}, flush=True)
        print('Original 65', {arm: v['score'] for arm, v in result['original_65'].items()}, flush=True)
    result.update({'adapters': metadata, 'inference': inference, 'policies': POLICIES, 'per_query': rows})
    if {'pretrained', 'context_expanded', 'full_plus_extractive_max'} & metadata.keys():
        result['current_list_policies'] = CURRENT_LIST_POLICIES
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
