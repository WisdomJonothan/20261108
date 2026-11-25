"""Learn final ranking weights from held-out-safe first-stage and evidence scores."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges, choose, summarize
from eval_rerank_headroom_study import diagnose
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import z
from replay_anchor_fallback import index
from replay_evidence_document_repair import fit, grade, label


FEATURES = ('pool_prior', 'pool_head_membership', 'dense_similarity', 'dense_log_discount',
            'frozen_score', 'frozen_membership', 'evidence_logit', 'bounded_evidence_logit',
            'outsider_evidence_logit')
METHODS = ('binary', 'pair_uniform', 'pair_cutoff')
ARMS = ('current',) + tuple(f'{method}/{mode}' for method in METHODS
                            for mode in ('free', 'protect_first', 'limited_edit'))
DISCOUNT = 1 / np.log2(2 + np.arange(30))


def materialize(manifest, paths, root, references, split, scope, excluded):
    fitted, metadata, _ = candidate_judges(paths)
    if set(fitted) != {'pretrained'}:
        raise ValueError('Expected logits from a single frozen evidence checkpoint')
    scores = fitted['pretrained']
    rows = [r for r in load(manifest)['queries'] if (r['domain'], r['qid']) not in excluded]
    keys = {(r['domain'], r['qid']) for r in rows}
    if len(rows) != len(keys) or scores.keys() != keys or {r['split'] for r in rows} != {split}:
        raise ValueError('The candidate manifests and score queries differ')
    group = defaultdict(list)
    for r in rows:
        group[r['domain']].append(r)
    current = {}
    if scope != 'fit':
        current = index({'per_query': references['oof_per_query' if split == 'train' else 'dev_per_query']})
        if keys != current.keys():
            raise ValueError('Expected the exact official or held-out reference query IDs')
    result = []
    for domain, records in sorted(group.items()):
        dense = read_trec_run(root / f'dense_mxbai_{domain}_{split}.trec')
        frozen = {}
        for fold in {r['cohort'] for r in records}:
            tag = 'sub_sft_e2bc' if split == 'dev' else f'trainfuse_{fold}'
            frozen[fold] = read_trec_run(root / f'{tag}_{domain}_{split}.trec')
        for row in records:
            qid, docs, fold = row['qid'], row['docs'], row['cohort']
            key = (domain, qid)
            logit = scores[key]['scores']
            if len(docs) != 30 or len(set(docs)) != 30 or set(docs) != logit.keys():
                raise ValueError(f'Unexpected candidate scores for {key}')
            dense_values = {d: (rank, value) for rank, value, d in dense[qid]}
            if not set(docs) <= dense_values.keys():
                raise ValueError(f'Documents missing in original first-stage scores for {key}')
            frozen_values = {d: v for _, v, d in frozen[fold][qid]}
            lower = min(frozen_values.values())
            inside = np.array([d in frozen_values for d in docs])
            candidate_values = [frozen_values.get(d, lower) for d in docs]
            judge = np.array([logit[d] for d in docs])
            head = np.arange(30) < 10
            matrix = np.column_stack((z(DISCOUNT), head, z([dense_values[d][1] for d in docs]),
                                      1/np.log2(1+np.array([dense_values[d][0] for d in docs])),
                                      z(candidate_values), inside, z(judge), np.tanh(judge / 5.),
                                      z(judge)*~head))
            reference = docs[:10]
            if scope != 'fit':
                policy = references['choices'][fold] if split == 'train' else references['selected']
                reference = current[key]['rankings'][policy]
            if not set(reference) <= set(docs):
                raise ValueError(f'Reference contains a document outside the matched list for {key}')
            result.append(row | {'features': matrix, 'current': reference})
    return result, metadata['pretrained']


def fit_pairs(rows, cutoff):
    domains = Counter(r['domain'] for r in rows)
    features, weights = [], []
    for r in rows:
        target = np.array([d in r['gold'] for d in r['docs']], dtype=int)
        pos, neg = np.nonzero(target[:, None] > target[None, :])
        if not len(pos):
            continue
        delta = r['features'][pos] - r['features'][neg]
        base_discount = DISCOUNT.copy()
        base_discount[10:] = 0
        magnitudes = np.abs(base_discount[pos] - base_discount[neg]) + .05 if cutoff else np.ones(len(pos))
        magnitudes /= magnitudes.sum()*domains[r['domain']]
        features.append(delta)
        weights.append(magnitudes)
    x = np.concatenate(features)
    w = np.tile(np.concatenate(weights), 2)
    w /= w.mean()
    y = np.repeat((1, 0), len(x))
    model = LogisticRegression(C=.3, fit_intercept=False, max_iter=600)
    return model.fit(np.concatenate((x, -x)), y, sample_weight=w)


def estimate(rows, models):
    predicted = []
    for row in rows:
        current, docs = row['current'], row['docs']
        rankings = {'current': current}
        for name, model in models.items():
            value = dict(zip(docs, model.decision_function(row['features']), strict=True))
            order = sorted(docs, key=lambda d: -value[d])
            rankings[f'{name}/free'] = order[:10]
            rankings[f'{name}/protect_first'] = [current[0], *[d for d in order if d != current[0]][:9]]
            result = current.copy()
            seen = set(current)
            incoming = next(d for d in order if d not in seen)
            slot = max(range(2, 10), key=lambda i: DISCOUNT[i] * (value[incoming]-value[current[i]]))
            if value[incoming]-value[current[slot]] >= .20:
                result[slot] = incoming
            rankings[f'{name}/limited_edit'] = result
        predicted.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'actual_reference')} |
                         {'rankings': rankings})
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-manifest', 'held-manifest', 'data-root', 'runs-root', 'repair', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--dev-manifest', type=Path)
    for name in ('training-scores', 'held-scores', 'dev-scores'):
        parser.add_argument(f'--{name}', type=Path, nargs='+', required=name != 'dev-scores')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    references = load(args.repair)
    held, identity = materialize(args.held_manifest, args.held_scores, args.runs_root, references,
                                 'train', 'held', set())
    excluded = {(r['domain'], r['qid']) for r in held}
    train, other = materialize(args.training_manifest, args.training_scores, args.runs_root, references,
                               'train', 'fit', excluded)
    if identity != other or len(held) != 186 or len(train) != 783:
        raise ValueError('Unexpected split size or model identity')
    train = label(train, 'train', args.data_root)
    models = {'binary': fit(train), 'pair_uniform': fit_pairs(train, False), 'pair_cutoff': fit_pairs(train, True)}
    rated = grade(estimate(held, models), label(held, 'train', args.data_root))
    by_fold = {f: choose([r for r in rated if r['cohort'] != f], ARMS) for f in {r['cohort'] for r in rated}}
    selected = choose(rated, ARMS)
    result = {'selected': selected, 'by_fold': by_fold, 'training': summarize(rated, selected, by_fold),
              'original_65': summarize([r for r in rated if r['actual_reference']], selected, by_fold),
              'fitted_on': len(train), 'feature_names': FEATURES, 'model_identity': identity,
              'coefficients': {k: model.coef_[0].tolist() for k, model in models.items()}, 'per_query': rated}
    print('Held-out selection', selected, by_fold, {a: v['score'] for a, v in result['training'].items()}, flush=True)
    if args.dev_manifest:
        if args.dev_scores is None:
            parser.error('Provide official-dev evidence scores')
        dev, other = materialize(args.dev_manifest, args.dev_scores, args.runs_root, references,
                                 'dev', 'dev', set())
        if len(dev) != 519 or other != identity:
            raise ValueError('Expected 519 official queries with the same evidence checkpoint')
        official = grade(estimate(dev, models), label(dev, 'dev', args.data_root))
        result.update({'dev': summarize(official, selected), 'held_per_query': rated, 'per_query': official,
                       'diagnostics': diagnose(official, selected, args.data_root, 'current')})
        if abs(result['dev']['current']['score'] - .3958942538410169) > 1e-12:
            raise ValueError('The official reference score changed')
        print('Official dev', {a: v['score'] for a, v in result['dev'].items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
