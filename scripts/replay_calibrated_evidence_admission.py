"""Calibrate relevance for incumbent and new candidates using fixed multi-model scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges, choose, summarize
from eval_rerank_headroom_study import diagnose
from replay_anchor_edit_scope import z
from replay_anchor_fallback import index
from replay_evidence_document_repair import DISCOUNT, fit, grade, label


POLICIES = {
    'admit30': (30, 0.03),
    'admit60': (60, 0.03),
    'admit60_guarded': (60, 0.07),
}
ARMS = ('current', 'reorder', *POLICIES, 'replace_and_reorder')


def prepare(manifest, repair, first, second, split):
    queries = index({'per_query': load(manifest)['queries']})
    report = load(repair)
    references = index({'per_query': report['oof_per_query' if split == 'train' else 'dev_per_query']})
    if queries.keys() != references.keys():
        raise ValueError('The candidate manifest and reference lists differ')
    inputs = []
    for paths in (first, second):
        scores, metadata, _ = candidate_judges(paths)
        if set(scores) != {'pretrained'} or scores['pretrained'].keys() != queries.keys():
            raise ValueError('Each pretrained checkpoint needs to score the full manifest')
        inputs.append((scores['pretrained'], metadata['pretrained']))
    rows = []
    for key, row in queries.items():
        documents = row['docs']
        if len(documents) != 60 or len(set(documents)) != 60:
            raise ValueError('Expected the same 60 unique candidates per query')
        policy = report['choices'][row['cohort']] if split == 'train' else report['selected']
        current = references[key]['rankings'][policy]
        current_positions = {d: DISCOUNT[i] for i, d in enumerate(current)}
        original = 1/np.log2(2+np.arange(60))
        features = [original, [current_positions.get(doc, 0.) for doc in documents]]
        features.extend(np.eye(4)[np.repeat(np.arange(4), [30, 15, 7, 8])].T[1:])
        models = []
        for raw, _ in inputs:
            scores = raw[key]['scores']
            if scores.keys() != set(documents):
                raise ValueError(f'Model scores differ from candidates for {key}')
            values = np.array([scores[d] for d in documents])
            norm = z(values)
            models.append(norm)
            features.extend((norm, np.tanh(values / 5.)))
        features.extend((np.minimum(*models), np.abs(models[0] - models[1])))
        rows.append(row | {'current': current, 'features': np.column_stack(features)})
    return rows, [metadata for _, metadata in inputs]


def one_admission(current, documents, probabilities, width, threshold):
    seen = set(current)
    challenger = max((d for d in documents[:width] if d not in seen), key=probabilities.__getitem__)
    pos = max(range(2, 10), key=lambda i: DISCOUNT[i] * (probabilities[challenger]-probabilities[current[i]]))
    revised = current.copy()
    if probabilities[challenger] - probabilities[current[pos]] > threshold:
        revised[pos] = challenger
    return revised


def predict(rows, model):
    result = []
    for row in rows:
        docs, current = row['docs'], row['current']
        values = model.predict_proba(row['features'])[:, 1]
        probability = dict(zip(docs, values, strict=True))
        ranks = {'current': current, 'reorder': [current[0], *sorted(current[1:], key=lambda d: -probability[d])]}
        for name, (width, threshold) in POLICIES.items():
            ranks[name] = one_admission(current, docs, probability, width, threshold)
        updated = ranks['admit60_guarded']
        ranks['replace_and_reorder'] = [updated[0], *sorted(updated[1:], key=lambda d: -probability[d])]
        result.append({key: row[key] for key in ('domain', 'qid', 'cohort', 'actual_reference')} | {'rankings': ranks})
    return result


def crossfit(rows):
    folds = sorted({r['cohort'] for r in rows})
    estimates, choices = [], {}
    for held in folds:
        training = [r for r in rows if r['cohort'] != held]
        test = [r for r in rows if r['cohort'] == held]
        inner = []
        for fold in (f for f in folds if f != held):
            left = [r for r in training if r['cohort'] == fold]
            model = fit([r for r in training if r['cohort'] != fold])
            inner.extend(grade(predict(left, model), left))
        choices[held] = choose(inner, ARMS)
        estimates.extend(grade(predict(test, fit(training)), test))
    return estimates, choices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('train-manifest', 'data-root', 'repair', 'output'):
        ap.add_argument(f'--{name}', type=Path, required=True)
    ap.add_argument('--dev-manifest', type=Path)
    for name in ('train-first', 'train-second', 'dev-first', 'dev-second'):
        ap.add_argument(f'--{name}', type=Path, nargs='+', required=name.startswith('train-'))
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a fresh output path')
    train, metadata = prepare(args.train_manifest, args.repair, args.train_first, args.train_second, 'train')
    if len(train) != 186:
        raise ValueError('Expected the fixed training holdout')
    train = label(train, 'train', args.data_root)
    held, choices = crossfit(train)
    selected = choose(held, ARMS)
    result = {'selected': selected, 'by_fold': choices, 'models': metadata,
              'training': summarize(held, selected, choices),
              'original_65': summarize([r for r in held if r['actual_reference']], selected, choices),
              'per_query': held}
    print('Selected', selected, choices, {a: v['score'] for a, v in result['training'].items()}, flush=True)
    if args.dev_manifest:
        if args.dev_first is None or args.dev_second is None:
            ap.error('Provide both official-dev score collections')
        dev, dev_metadata = prepare(args.dev_manifest, args.repair, args.dev_first, args.dev_second, 'dev')
        if len(dev) != 519 or dev_metadata != metadata:
            raise ValueError('Official evaluation needs the same model checkpoints and the full dev set')
        estimates = grade(predict(dev, fit(train)), label(dev, 'dev', args.data_root))
        result.update({'dev': summarize(estimates, selected), 'per_query': estimates,
                       'heldout_per_query': held, 'diagnostics': diagnose(estimates, selected, args.data_root, 'current')})
        if abs(result['dev']['current']['score'] - .3958942538410169) > 1e-12:
            raise ValueError('Official reference score changed')
        print('Official dev', {a: v['score'] for a, v in result['dev'].items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
