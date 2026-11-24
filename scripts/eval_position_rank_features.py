"""Fit a small within-top-ten rank model on training folds and evaluate transfer."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from benchmark_pairwise_replacement_v2 import domain_macro, ndcg10, paired_bootstrap, read_qrels
from eval_position_repair import DEV_CE, read_json, select_internal
from lib_reteco import read_trec_run


def build(root, fold):
    split = 'dev' if fold == 'dev' else 'train'
    source = 'dev' if fold == 'dev' else f'train_{fold}'
    manifest = read_json(root / 'results/replacement_v2' / f'candidates_{source}.json')
    result = []
    for domain in sorted({r['domain'] for r in manifest['queries']}):
        tag = DEV_CE if fold == 'dev' else f'{fold}_ce'
        ce = read_trec_run(root / 'runs' / f'{tag}_{domain}_{split}_ce.trec')
        dense = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_{split}.trec')
        qrels = read_qrels(root / 'data/track1_tempo' / domain / f'qrels_{split}.txt')
        for row in (r for r in manifest['queries'] if r['domain'] == domain):
            qid, top = row['qid'], row['frozen_top10']
            dense_rank = {doc: rank for rank, _, doc in dense[qid]}
            ce_rank = {doc: rank for rank, _, doc in ce[qid]}
            ce_score = {doc: score for _, score, doc in ce[qid]}
            ranks = [(i + 1, dense_rank[doc], ce_rank[doc]) for i, doc in enumerate(top)]
            features = 1 / np.log2(1 + np.asarray(ranks, dtype=float))
            result.append({'qid': qid, 'domain': domain, 'top': top, 'features': features,
                           'gold': qrels[qid], 'actions': select_internal(top, ce_score)})
    return result


def fit(rows):
    differences, weights = [], []
    counts = Counter(row['domain'] for row in rows)
    for row in rows:
        positive = np.array([doc in row['gold'] for doc in row['top']])
        examples = row['features'][positive, None] - row['features'][None, ~positive]
        examples = examples.reshape(-1, 3)
        if len(examples):
            differences.extend(examples)
            weights.extend([1 / (len(examples) * counts[row['domain']])] * len(examples))
    x, w = np.asarray(differences), np.asarray(weights)
    w /= w.sum()

    def objective(coef):
        scores = x @ coef
        return (w @ np.logaddexp(0., -scores) + .005 * (coef @ coef),
                -(w * expit(-scores)) @ x + .01 * coef)

    learned = minimize(objective, np.zeros(3), method='L-BFGS-B', jac=True)
    if not learned.success:
        raise RuntimeError(learned.message)
    return learned.x


def predict(rows, coef):
    output = []
    for row in rows:
        top, gold = row['top'], row['gold']
        base = ndcg10(top, gold)
        scores = row['features'] @ coef
        ranking = [top[i] for i in np.argsort(-scores, kind='stable')]
        changed = top.copy()
        best_gain = 0.
        for i, j in row['actions']:
            gain = (scores[j] - scores[i]) * (1 / math.log2(i + 2) - 1 / math.log2(j + 2))
            if gain > best_gain:
                best_gain = gain
                changed = top.copy()
                changed[i], changed[j] = changed[j], changed[i]
        output.append({'qid': row['qid'], 'domain': row['domain'], 'baseline': base,
                       'one_exchange': ndcg10(changed, gold),
                       'full_reorder': ndcg10(ranking, gold)})
    return output


def summarize(rows):
    summary = {}
    for policy in ('one_exchange', 'full_reorder'):
        deltas = [{**r, 'delta': r[policy] - r['baseline']} for r in rows]
        summary[policy] = {'ndcg': domain_macro(rows, key=policy),
                           'delta': domain_macro(deltas, key='delta'),
                           'bootstrap': paired_bootstrap(deltas)}
    return {'queries': len(rows), 'baseline': domain_macro(rows, key='baseline'), 'arms': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folds = {name: build(args.root, name) for name in ('f0', 'f1', 'f2')}
    oof, weights = [], {}
    for held, test in folds.items():
        train = [r for key, rows in folds.items() if key != held for r in rows]
        assert not {(r['domain'], r['qid']) for r in test} & {(r['domain'], r['qid']) for r in train}
        coef = fit(train)
        weights[held] = coef.tolist()
        oof.extend(predict(test, coef))
    coef = fit([r for rows in folds.values() for r in rows])
    dev = predict(build(args.root, 'dev'), coef)
    assert len(oof) == 969 and len(dev) == 519
    report = {'features': ['frozen reciprocal log rank', 'dense reciprocal log rank',
                           'CE reciprocal log rank'],
              'protocol': 'Fixed L2 pairwise logistic fit, domain/query weight for informative pairs. '
              'Train-fold ranker scores are out of fold. The training holdout excludes 242 queries. '
              'Official dev uses only the final training-fold coefficient vector, with the fixed '
              'dev comparator CE and submission base ranking.',
              'coefficients': weights | {'all_train_folds': coef.tolist()},
              'summary': {'oof': summarize(oof), 'dev': summarize(dev)},
              'per_query': {'oof': oof, 'dev': dev}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    main()
