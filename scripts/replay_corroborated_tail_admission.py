"""Compare one-slot repairs using agreement between cached judges."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from replay_anchor_fusion_calibration import build
from replay_redundancy_accumulation import rerank


BASELINE = 'reference'
# (Evidence for replacement, permitted drop in existing blended score, earliest slot.)
ARMS = {
    'judge_close': ('judge', .15, 5),
    'judge_wide': ('judge', .35, 5),
    'specialist_close': ('specialist', .15, 5),
    'specialist_wide': ('specialist', .35, 5),
    'agreement_close': ('agreement', .15, 5),
    'agreement_wide': ('agreement', .35, 5),
    'agreement_last_two': ('agreement', .35, 8),
    'agreement_redundant': ('redundant', .35, 5),
    'judge_last_two': ('judge', .35, 8),
    'specialist_last_two': ('specialist', .35, 8),
}


def replace(order, blended, judge, specialist, sim, mode, margin, floor):
    outsiders = [i for i in range(len(blended)) if i not in order]
    best = None
    prompt_docs = set()
    for position in range(floor, len(order)):
        old = order[position]
        retained = [i for i in order if i != old]
        if mode == 'redundant' and sim[old, retained].max() < .8:
            continue
        for new in outsiders:
            penalty = sim[new, retained].max()
            if blended[new] < blended[old] - margin or penalty >= .8:
                continue
            j = judge[new] - judge[old]
            if mode != 'specialist' and j <= 0:
                continue
            if mode != 'judge':
                prompt_docs.update((old, new))
            s = specialist[new] - specialist[old]
            if mode != 'judge' and s <= 0:
                continue
            gain = j if mode == 'judge' else s if mode == 'specialist' else min(j, s)
            option = (gain, -position, -new)
            if best is None or option > best:
                best = option
    result = order.copy()
    if best is not None:
        result[-best[1]] = -best[2]
    return result, len(prompt_docs)


def predict(rows, protected, reference, duplicate, specialist):
    results = []
    fallback = set()
    for row in rows:
        key = row['domain'], row['qid']
        prompts = {}
        screened = {}
        proposals = {}
        if row['route'] == 'accepted':
            expected = protected[key]['rankings']['1']
            orders = dict.fromkeys((BASELINE, *ARMS), expected)
        else:
            fallback.add(key)
            docs = row['docs']
            locations = {doc: i for i, doc in enumerate(docs)}
            frozen, judge = row['features'][:, :2].T
            blend = .7 * frozen + .3 * judge
            raw = specialist[key]['scores']
            if raw.keys() != locations.keys():
                raise ValueError(f'Specialist scores do not match the 34 candidates for {key}')
            specialist_z = z([raw[doc] for doc in docs])
            similarity = np.zeros((len(docs), len(docs)))
            for a, b, value in duplicate[key]['near_duplicate_edges']:
                i, j = locations[a], locations[b]
                similarity[i, j] = similarity[j, i] = value
            expected = reference[key]['rankings']['exact_boost_2']
            baseline = rerank(docs, blend, similarity, 'exact', 2.)
            if baseline != expected:
                raise ValueError(f'Baseline differs for {key}')
            base = [locations[doc] for doc in baseline]
            orders = {BASELINE: baseline}
            for arm, params in ARMS.items():
                chosen, screened[arm] = replace(base, blend, judge, specialist_z, similarity, *params)
                # Agreement's standardized margin uses every 4B score in the pool.
                prompts[arm] = len(docs) if screened[arm] else 0
                orders[arm] = [docs[i] for i in chosen]
                proposals[arm] = any(a != b for a, b in zip(base, chosen))
        if any(len(set(v)) != 10 or v[0] != expected[0] for v in orders.values()):
            raise ValueError(f'Top-ten invariant failed for {key}')
        results.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'route')} |
                       {'rankings': orders, 'scores': {k: ndcg10(v, row['gold']) for k, v in orders.items()},
                        'extra_prompts': prompts, 'eligible_specialist_docs': screened, 'edits': proposals})
    if fallback != reference.keys() or fallback != specialist.keys():
        raise ValueError('The cached fallback query populations disagree')
    return results


def choose(rows):
    return max((BASELINE, *ARMS), key=lambda arm: macro((r['domain'], r['scores'][arm]) for r in rows))


def summarize(rows, selected, crossfit=None):
    choices = {arm: [arm] * len(rows) for arm in (BASELINE, *ARMS)}
    choices['selected'] = [crossfit[r['cohort']] if crossfit else selected for r in rows]
    result = {}
    for name, arms in choices.items():
        paired = paired_delta([(r['domain'], r['scores'][arm] - r['scores'][BASELINE])
                               for r, arm in zip(rows, arms)])
        result[name] = {
            'ndcg': macro((r['domain'], r['scores'][arm]) for r, arm in zip(rows, arms)),
            'change': paired,
            'edited_queries': sum(r['edits'].get(arm, False) for r, arm in zip(rows, arms)),
            'extra_prompts_per_query': sum(r['extra_prompts'].get(arm, 0) for r, arm in zip(rows, arms)) / len(rows),
            'eligible_specialist_docs_per_query': sum(r['eligible_specialist_docs'].get(arm, 0)
                                                     for r, arm in zip(rows, arms)) / len(rows),
        }
    return result


def matched_controls(rows):
    return {
        arm: paired_delta([(r['domain'], r['scores']['agreement_last_two'] - r['scores'][arm])
                           for r in rows])
        for arm in ('judge_last_two', 'specialist_last_two')
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    analysis = args.root / 'results' / 'analysis'

    def path(stem):
        return analysis / f'{stem}_20261125.json'

    budgeted = load(path('budgeted_anchor_fallback'))
    protected = load(path('accepted_weight_full_dev_audit'))
    duplicate = index(load(path('shortlist_near_duplicates')))
    reference = load(path('redundancy_stronger_replay'))
    specialists = load(path('specialist_shortlist_scores'))['per_query']
    cohorts = [path('anchor_reranker_qwen8b_sample26'), path('anchor_reranker_qwen8b_replication39')]
    train = build(budgeted, cohorts, path('anchor_edit_scores_oof100'), args.data_root, 'oof')

    def replay(rows, split, label):
        specialist = index({'per_query': [r for r in specialists if r['split'] == label]})
        return predict(rows, index(protected[split]), index(reference[split]), duplicate, specialist)

    training = replay(train, 'oof', 'train')
    selected = choose(training)
    crossfit = {i: choose([r for r in training if r['cohort'] != i]) for i in (0, 1)}
    print('Training choices:', selected, crossfit, flush=True)
    evaluated = build(budgeted, [path('anchor_reranker_qwen8b_dev100')],
                      path('anchor_edit_scores_dev100'), args.data_root, 'dev')
    dev = replay(evaluated, 'dev', 'dev')
    out = {'candidate_count': 34, 'fixed_8b_prompts_per_dev_query': 54.43,
           'arms': ARMS, 'pooled_choice': selected, 'choice_by_held_cohort': crossfit,
           'oof': summarize(training, selected, crossfit), 'dev': summarize(dev, selected),
           'matched_tail_controls': {'oof': matched_controls(training), 'dev': matched_controls(dev)},
           'oof_per_query': training, 'dev_per_query': dev}
    for split in ('oof', 'dev'):
        if abs(out[split][BASELINE]['ndcg'] - reference[split]['arms']['exact_boost_2']['ndcg']) > 1e-12:
            raise ValueError(f'Full-system reference score differs on {split}')
        for arm, result in out[split].items():
            print(split, arm, result['ndcg'], result['change']['ci95'], result['edited_queries'],
                  result['extra_prompts_per_query'], flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(out, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
