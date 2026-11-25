"""Calibrate the relative contribution of pointwise scores and verified evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges, choose, grade, summarize
from replay_anchor_cap import z
from replay_anchor_fallback import index
from replay_evidence_document_repair import fit, label


FEATURE_SETS = ('pointwise', 'evidence', 'joint')
ARMS = ('verified', *FEATURE_SETS)


def prepare(manifest, observations, first, second):
    queries = load(manifest)['queries']
    cache = index({'per_query': [r for path in observations for r in load(path)['per_query']]})
    first, second = (candidate_judges(paths)[0]['pretrained'] for paths in (first, second))
    if len(cache) != len(queries) or not (cache.keys() == first.keys() == second.keys()):
        raise ValueError('Each input must cover the exact evaluation queries')
    result = []
    for query in queries:
        key = query['domain'], query['qid']
        docs = query['docs']
        obs = cache[key]['observations']
        if len(set(docs)) != 14 or obs.keys() != set(docs):
            raise ValueError(f'Expected the same 14 documents for {key}')
        scores = [np.array([model[key]['scores'][doc] for doc in docs]) for model in (first, second)]
        direct, reasoned, verified = (np.array([obs[doc][name] for doc in docs])
                                      for name in ('direct', 'reasoned', 'verified'))
        checked = np.where(verified, reasoned, direct)
        prior = z(1 / np.log2(2 + np.arange(len(docs))))
        always = [prior, np.arange(len(docs)) < 10, np.arange(len(docs)) >= 12]
        evidence = [direct / 3, (checked - direct) / 3, z(checked), verified]
        individual = [f for x in scores for f in (z(x), np.tanh(x / 5))]
        features = {'pointwise': np.column_stack(always + individual),
                    'evidence': np.column_stack(always + evidence),
                    'joint': np.column_stack(always + evidence + individual)}
        keys = ('domain', 'qid', 'cohort', 'split', 'actual_reference', 'docs')
        result.append({k: query[k] for k in keys} | {'x': features, 'prior': prior, 'checked': z(checked)})
    return result


def trained(rows):
    return {source: fit([r | {'features': r['x'][source]} for r in rows]) for source in FEATURE_SETS}


def estimate(rows, models, data):
    orders = []
    for r in rows:
        docs = r['docs']
        values = {'verified': r['checked']}
        for source, model in models.items():
            values[source] = z(model.decision_function(r['x'][source]))
        rankings = {'current': docs[:10]}
        for source, scores in values.items():
            order = np.argsort(-(r['prior'] + scores), kind='stable')
            rankings[source] = [docs[i] for i in order[:10]]
        orders.append({k: r[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                      {'rankings': rankings})
    return grade(orders, rows[0]['split'], data)


def crossfit(rows, data):
    folds = sorted({r['cohort'] for r in rows})
    by_fold, estimates = {}, []
    for fold in folds:
        train = [r for r in rows if r['cohort'] != fold]
        test = [r for r in rows if r['cohort'] == fold]
        inner = []
        for nested in (f for f in folds if f != fold):
            inner.extend(estimate([r for r in train if r['cohort'] == nested],
                                  trained([r for r in train if r['cohort'] != nested]), data))
        by_fold[fold] = choose(inner, ARMS)
        estimates.extend(estimate(test, trained(train), data))
    return estimates, by_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-root', 'output', 'dev-manifest', 'selection'):
        parser.add_argument(f'--{name}', type=Path, required=not name.startswith(('dev-', 'selection')))
    for name in ('observations', 'first', 'second', 'dev-observations', 'dev-first', 'dev-second'):
        parser.add_argument(f'--{name}', type=Path, nargs='+', required=not name.startswith('dev-'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')

    rows = label(prepare(args.manifest, args.observations, args.first, args.second), 'train', args.data_root)
    if len(rows) != 186:
        raise ValueError('Expected the fixed 186-query holdout')
    if args.dev_manifest:
        if not all((args.selection, args.dev_observations, args.dev_first, args.dev_second)):
            parser.error('Official evaluation requires previously selected policy and all dev inputs')
        selected = load(args.selection)['selected']
        dev = prepare(args.dev_manifest, args.dev_observations, args.dev_first, args.dev_second)
        if len(dev) != 519 or selected not in ARMS:
            raise ValueError('Unexpected official-dev population or ranking choice')
        ranks = estimate(dev, trained(rows), args.data_root)
        summary = summarize(ranks, selected)
        report = {'dev': summary}
    else:
        ranks, by_fold = crossfit(rows, args.data_root)
        selected = choose(ranks, ARMS)
        summary = summarize(ranks, selected, by_fold)
        report = {'training': summary, 'by_fold': by_fold,
                  'original_65': summarize([r for r in ranks if r['actual_reference']], selected, by_fold)}
    report.update({'selected': selected, 'per_query': ranks})
    print(selected, {arm: v['score'] for arm, v in summary.items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
