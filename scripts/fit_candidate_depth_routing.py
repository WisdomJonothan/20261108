"""Fit candidate-depth priorities on train folds and compare at matched budgets."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from audit_adaptive_candidate_depth import route, summarize
from compare_anchor_reference import load
from coverage_cf_bound import read_qrels
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


FOLDS = ('f0', 'f1', 'f2')
FEATURES = ('dense_margin', 'dense_head_gap', 'top_ten_overlap', 'fused_dense_rank_mean', 'abstained')
DISCOUNTS = [0.0] + list(np.cumsum([1 / math.log2(k + 1) for k in range(1, 11)]))


def rows_for_split(root, runs, data, split, reference):
    folder = root / 'results/replacement_v2'
    folds = FOLDS if split == 'train' else ('dev',)
    items = [
        row | {'fold': fold}
        for fold in folds
        for row in load(folder / ('candidates_dev.json' if fold == 'dev' else f'candidates_train_{fold}.json'))['queries']
    ]
    sample = index({'per_query': items})
    policies = index(load(reference))
    if sample.keys() != policies.keys() or len(sample) != (969 if split == 'train' else 519):
        raise ValueError(f'Unexpected query membership for {split}')
    groups = defaultdict(list)
    for domain in sorted({d for d, _ in sample}):
        run = read_trec_run(runs / f'dense_mxbai_{domain}_{split}.trec')
        labels = read_qrels(data / domain / f'qrels_{split}.txt')
        for key, item in sample.items():
            if key[0] != domain:
                continue
            qid = key[1]
            docs = [doc for _, _, doc in run[qid][:300]]
            expected = item['frozen_top10']
            if len(set(docs)) != 300 or not set(expected) <= set(docs[:100]):
                raise ValueError(f'Unexpected dense pool for {key}')
            ref = policies[key]
            action = ref['policies']['external6']
            if choice := action['chosen']:
                if expected[choice['slot'] - 1] != choice['incumbent'] or choice['challenger'] not in docs[:100]:
                    raise ValueError(f'Competition action cannot be reconstructed for {key}')
            saved = labels[qid]
            a, b = len(saved & set(docs[:100])), len(saved & set(docs))
            ideal = DISCOUNTS[min(10, len(saved))]
            scores = [score for _, score, _ in run[qid][:100]]
            scale = max(pstdev(scores), 1e-8)
            ranks = {doc: rank for rank, doc in enumerate(docs[:100], 1)}
            groups[domain].append({
                'domain': domain, 'qid': qid, 'fold': item['fold'],
                'abstained': choice is None, 'oracle100': float(DISCOUNTS[min(a, 10)] / ideal) if ideal else 0.,
                'oracle300': float(DISCOUNTS[min(b, 10)] / ideal) if ideal else 0.,
                'newly_reachable': a == 0 and b > 0, 'competition_score': ref['baseline'] + action['delta'],
                'dense_margin': (scores[0] - scores[9]) / scale,
                'dense_head_gap': (mean(scores[:10]) - mean(scores[10:])) / scale,
                'top_ten_overlap': len(set(expected) & set(docs[:10])),
                'fused_dense_rank_mean': mean(ranks[doc] for doc in expected),
            })
    return groups


def fit_predict(train, evaluation):
    def features(rows):
        return [[float(r[f]) for f in FEATURES] for r in rows]

    train_domains = Counter(r['domain'] for r in train)
    weights = [len(train) / (len(train_domains) * train_domains[r['domain']]) for r in train]
    scaler = StandardScaler().fit(features(train))
    x_train = scaler.transform(features(train))
    x_eval = scaler.transform(features(evaluation))
    gain = [r['oracle300'] - r['oracle100'] for r in train]
    reachable = [r['newly_reachable'] for r in train]
    model_gain = Ridge(alpha=10.0).fit(x_train, gain, sample_weight=weights)
    model_reach = LogisticRegression(C=1.0, max_iter=2000).fit(x_train, reachable, sample_weight=weights)
    for row, g, z in zip(evaluation, model_gain.predict(x_eval), model_reach.predict_proba(x_eval)[:, 1]):
        row['predicted'] = {'ridge_gain': float(g), 'logistic_reach': float(z)}


def report(groups):
    rows = route(groups)
    for items in groups.values():
        capacity = sum(r['abstained'] for r in items)
        for model in ('ridge_gain', 'logistic_reach'):
            order = sorted(items, key=lambda r: (-r['predicted'][model], r['qid']))
            for i, row in enumerate(order):
                row['expanded'][model] = float(i < capacity)
    paired = {
        model: paired_delta([
            (r['domain'], (r['oracle300'] - r['oracle100']) *
             (r['expanded'][model] - r['expanded']['rank_disagreement'])) for r in rows
        ]) for model in ('ridge_gain', 'logistic_reach')
    }
    return {
        'queries': len(rows), 'domains': len(groups),
        'newly_reachable_at_300': sum(r['newly_reachable'] for r in rows),
        'depth100_ideal_ndcg': mean(mean(r['oracle100'] for r in items) for items in groups.values()),
        'competition_ndcg': mean(mean(r['competition_score'] for r in items) for items in groups.values()),
        'policies': summarize(rows), 'learned_vs_disagreement': paired, 'per_query': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyses = args.root / 'results/analysis'
    train = rows_for_split(args.root, args.runs_root, args.data_root, 'train',
                           analyses / 'position_oracle_oof_20261124.json')
    dev = rows_for_split(args.root, args.runs_root, args.data_root, 'dev',
                         analyses / 'position_oracle_dev_20261124.json')
    training_rows = [r for rows in train.values() for r in rows]
    official_rows = [r for rows in dev.values() for r in rows]
    if {(r['domain'], r['qid']) for r in training_rows} & {(r['domain'], r['qid']) for r in official_rows}:
        raise ValueError('Training and evaluation query membership overlap')
    for fold in FOLDS:
        fit_predict([r for r in training_rows if r['fold'] != fold],
                    [r for r in training_rows if r['fold'] == fold])
    fit_predict(training_rows, official_rows)
    result = {
        'protocol': 'Cross-fit two fixed supervised predictors on independent training folds, then fit '
                    'on pooled training folds for official dev. Fit transformations within each training '
                    'sample. Ridge predicts the ideal nDCG change. Logistic regression predicts a first '
                    'positive recovered between depth 100 and 300. Weigh domains equally in fitting. '
                    'All selective policies expand the same number of queries per domain as abstention.',
        'features': FEATURES, 'oof': report(train), 'dev': report(dev),
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, separators=(',', ':'))
        stream.write('\n')
    for split in ('oof', 'dev'):
        print(split, result[split]['queries'], result[split]['newly_reachable_at_300'], flush=True)
        for name, item in result[split]['policies'].items():
            print(name, round(item['delta_ideal_ndcg']['point'], 6), item['newly_reachable_zeroes'])
        for name, compare in result[split]['learned_vs_disagreement'].items():
            print(name, 'versus rank disagreement', compare['point'], compare['ci95'])


if __name__ == '__main__':
    main()
