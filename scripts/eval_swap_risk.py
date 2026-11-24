"""Cross-fit separate beneficial and harmful swap probabilities at six pairs."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from coverage_cf_bound import DOMAINS, read_qrels
from lib_reteco import read_trec_run
from replay_position_discount import bootstrap, eligible


FEATURES = ('slot discount', 'prior', 'mean challenger logit', 'min challenger logit',
            'order agreement', 'incumbent log dense rank', 'challenger log dense rank',
            'challenger appearances among six')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def observations(root, data, run_root, split):
    file = root / 'results/analysis' / f'position_oracle_{split}_20261124.json'
    rows = load(file)['per_query']
    fold_of = load(root / 'results/replacement_v2/ce_folds.json')['fold_of']
    partition = 'train' if split == 'oof' else 'dev'
    for domain in DOMAINS:
        query_rows = [r for r in rows if r['domain'] == domain]
        dense = read_trec_run(run_root / f'dense_mxbai_{domain}_{partition}.trec')
        gold = read_qrels(data / domain / f'qrels_{partition}.txt')
        for row in query_rows:
            qid = row['qid']
            ranked = {doc: rank for rank, _, doc in dense[qid]}
            targets = gold[qid]
            ideal = sum(1 / math.log2(i + 2) for i in range(min(len(targets), 10)))
            actions = row['actions']['external']
            assert len(actions) == 6
            frequency = Counter(a['challenger'] for a in actions)
            for a in actions:
                discount = 1 / math.log2(a['slot'] + 1)
                logs = np.clip([a['p_fwd'], a['p_rev']], 1e-5, 1 - 1e-5)
                logs = np.log(logs / (1 - logs))
                k, c = a['incumbent'], a['challenger']
                raw_effect = int(c in targets) - int(k in targets)
                assert abs(a['delta'] - raw_effect * discount / ideal) < 1e-10
                a['features'] = [discount, a['prior'], float(logs.mean()), float(logs.min()),
                                 int(a['agree']), math.log1p(ranked.get(k, 1001)),
                                 math.log1p(ranked.get(c, 1001)), frequency[c]]
                a['target'] = int(k in targets) * 2 + int(c in targets)
            row['fold'] = fold_of[qid] if split == 'oof' else None
    return rows


def matrix(rows):
    actions = [a for r in rows for a in r['actions']['external']]
    return np.asarray([a['features'] for a in actions], dtype=float)


def fit(rows):
    actions = [a for row in rows for a in row['actions']['external']]
    x = matrix(rows)
    counts = Counter(row['domain'] for row in rows)
    weights = np.array([1 / counts[row['domain']] for row in rows for _ in row['actions']['external']])
    weights /= weights.sum()
    mu, sd = weights @ x, np.sqrt(weights @ ((x - weights @ x) ** 2)).clip(1e-6)
    v = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    y = np.array([a['target'] for a in actions])
    onehot = np.eye(4)[y]
    reg = .01

    def objective(w):
        coeff = w.reshape(-1, 4)
        logits = v @ coeff
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        score = -weights @ logp[np.arange(len(y)), y] + reg * np.sum(coeff[1:] ** 2) / 2
        gradient = v.T @ ((np.exp(logp) - onehot) * weights[:, None])
        gradient[1:] += reg * coeff[1:]
        return score, gradient.ravel()

    result = minimize(objective, np.zeros(v.shape[1] * 4), method='L-BFGS-B', jac=True)
    if not result.success:
        raise RuntimeError(result.message)
    direct = np.array([a['delta'] for a in actions])
    penalty = np.eye(v.shape[1]) * reg
    penalty[0, 0] = 0.
    ridge = np.linalg.solve(v.T @ (weights[:, None] * v) + penalty, v.T @ (weights * direct))
    return mu, sd, result.x.reshape(-1, 4), ridge


def select(rows, model):
    mu, sd, coef, ridge = model
    output = []
    for row in rows:
        actions = row['actions']['external']
        v = np.column_stack([np.ones(len(actions)), (matrix([row]) - mu) / sd])
        logits = v @ coef
        probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        discount = np.asarray([a['features'][0] for a in actions])
        old = max(eligible(actions), key=lambda a: a['mp'], default=None)
        assert abs((old['delta'] if old else 0.) - row['policies']['external6']['delta']) < 1e-12
        result = {k: row[k] for k in ('domain', 'qid', 'baseline', 'fold')}
        result['fixed'] = {'delta': old['delta'] if old else 0., 'chosen': old is not None}
        policies = {'ridge_control': v @ ridge,
                    'factorized': discount * (probabilities[:, 1] - probabilities[:, 2]),
                    'factorized_harm2': discount * (probabilities[:, 1] - 2 * probabilities[:, 2])}
        for name, ranking in policies.items():
            i = int(ranking.argmax())
            chosen = ranking[i] > 0
            result[name] = {'delta': actions[i]['delta'] if chosen else 0., 'chosen': bool(chosen),
                            'slot': actions[i]['slot'] if chosen else None,
                            'p_help': float(probabilities[i, 1]) if chosen else None,
                            'p_harm': float(probabilities[i, 2]) if chosen else None}
        output.append(result)
    return output


def summarize(rows):
    base = bootstrap([(r['domain'], r['baseline']) for r in rows])['point']
    summary = {}
    for name in ('fixed', 'ridge_control', 'factorized', 'factorized_harm2'):
        changes = [(r['domain'], r[name]['delta'] - r['fixed']['delta']) for r in rows]
        delta = bootstrap([(r['domain'], r[name]['delta']) for r in rows])['point']
        counts = Counter('kept' if not r[name]['chosen'] else 'helpful' if r[name]['delta'] > 1e-12
                         else 'harmful' if r[name]['delta'] < -1e-12 else 'neutral' for r in rows)
        summary[name] = {'ndcg': base + delta, 'gain_vs_base': delta,
                         'vs_fixed': bootstrap(changes), 'counts': counts,
                         'zero_queries_recovered': sum(r['baseline'] <= 1e-12 and r[name]['delta'] > 1e-12
                                                       for r in rows)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    train = observations(args.root, args.data_root, args.run_root, 'oof')
    dev = observations(args.root, args.data_root, args.run_root, 'dev')
    assert len(train) == 969 and len(dev) == 519
    assert not {(r['domain'], r['qid']) for r in train} & {(r['domain'], r['qid']) for r in dev}
    crossfit = []
    for fold in (0, 1, 2):
        fitted = fit([r for r in train if r['fold'] != fold])
        crossfit.extend(select([r for r in train if r['fold'] == fold], fitted))
    assert len(crossfit) == len(train)
    predictions = select(dev, fit(train))
    report = {'protocol': 'Fixed six external actions. Fit on folds only. Four outcomes encode both '
              'documents’ binary relevance and separate helpful and harmful probabilities. '
              'Choose a positive expected discounted effect, with a prespecified double-harm control. '
              'Compare to direct ridge with the same inputs. Qrels score the dev outputs only.',
              'features': FEATURES, 'summary': {'oof': summarize(crossfit), 'dev': summarize(predictions)},
              'per_query': {'oof': crossfit, 'dev': predictions}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    main()
