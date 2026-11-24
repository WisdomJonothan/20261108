"""Fit a challenger relevance prior on training folds; measure candidate coverage."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from coverage_cf_bound import DOMAINS, ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_position_discount import bootstrap


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def z(scores):
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.mean()) / max(float(scores.std()), 1e-8)


def build(root, fold):
    dev = fold == 'dev'
    split = 'dev' if dev else 'train'
    scope = 'dev' if dev else f'train_{fold}'
    ce_tag = 'fusion100_full_ce_s17_20261115' if dev else f'{fold}_ce'
    records = load(root / 'results/replacement_v2' / f'candidates_{scope}.json')['queries']
    output = []
    for domain in DOMAINS:
        dense = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_{split}.trec')
        ce = read_trec_run(root / 'runs' / f'{ce_tag}_{domain}_{split}_ce.trec')
        gold = read_qrels(root / 'data/track1_tempo' / domain / f'qrels_{split}.txt')
        for row in (r for r in records if r['domain'] == domain):
            qid = row['qid']
            top = row['frozen_top10']
            top_set = set(top)
            docs = [doc for _, _, doc in dense[qid][:100]]
            ranked = {doc: (rank, score) for rank, score, doc in ce[qid]}
            assert top_set <= set(docs) == set(ranked) and len(ranked) == 100
            dr = np.arange(1, 101)
            cr = np.array([ranked[doc][0] for doc in docs])
            dz = z([score for _, score, _ in dense[qid][:100]])
            cz = z([ranked[doc][1] for doc in docs])
            reciprocal_dense, reciprocal_ce = 1 / np.log2(dr + 1), 1 / np.log2(cr + 1)
            x = np.column_stack((dz, cz, reciprocal_dense, reciprocal_ce,
                                 np.abs(dz - cz), np.minimum(dz, cz)))
            y = np.array([doc in gold[qid] for doc in docs], dtype=float)
            reference = np.lexsort((dr, cr < dr, np.minimum(dr, cr)))
            outside = np.array([doc not in top_set for doc in docs])
            output.append({'qid': qid, 'domain': domain, 'fold': fold, 'docs': docs,
                           'x': x, 'gold': y, 'ref': reference, 'outside': outside,
                           'baseline': ndcg10(top, gold[qid])})
    assert len(output) == len(records)
    return output


def fit(rows):
    counts = Counter(r['domain'] for r in rows)
    x = np.concatenate([r['x'][r['outside']] for r in rows])
    y = np.concatenate([r['gold'][r['outside']] for r in rows])
    w = np.concatenate([np.full(r['outside'].sum(), 1 / counts[r['domain']] / r['outside'].sum())
                        for r in rows])
    w /= w.sum()
    mu = w @ x
    sd = np.sqrt(w @ (x - mu) ** 2).clip(1e-6)
    v = np.column_stack((np.ones(len(x)), (x - mu) / sd))

    def loss(coef):
        scores = v @ coef
        reg = .001
        gradient = v.T @ (w * (expit(scores) - y))
        gradient[1:] += reg * coef[1:]
        return w @ (np.logaddexp(0, scores) - y * scores) + .5 * reg * (coef[1:] @ coef[1:]), gradient

    result = minimize(loss, np.zeros(v.shape[1]), method='L-BFGS-B', jac=True)
    if not result.success:
        raise RuntimeError(result.message)
    return mu, sd, result.x


def predict(rows, params):
    mu, sd, coeff = params
    output = []
    for row in rows:
        prediction = ((row['x'] - mu) / sd) @ coeff[1:] + coeff[0]
        learned = np.argsort(-prediction, kind='stable')
        entry = {k: row[k] for k in ('qid', 'domain', 'fold', 'baseline')}
        entry['reachable_zero'] = row['baseline'] <= 1e-12 and bool(row['gold'].any())
        for name, order in [('rank_union', row['ref']), ('supervised', learned)]:
            selected = order[row['outside'][order]][:20]
            assert len(selected) == 20
            entry[name] = {'hit6': bool(row['gold'][selected[:6]].any()),
                           'hit20': bool(row['gold'][selected].any()),
                           'six': [row['docs'][i] for i in selected[:6]]}
        output.append(entry)
    return output


def summarize(rows):
    zero = [r for r in rows if r['reachable_zero']]
    out = {'queries': len(rows), 'reachable_zero_queries': len(zero), 'all': {}, 'reachable_zero': {}}
    for key, part in [('all', rows), ('reachable_zero', zero)]:
        for metric in ('hit6', 'hit20'):
            diff = [(r['domain'], int(r['supervised'][metric]) - int(r['rank_union'][metric])) for r in part]
            out[key][metric] = {'rank_union': sum(r['rank_union'][metric] for r in part),
                                'supervised': sum(r['supervised'][metric] for r in part),
                                'paired_macro_delta': bootstrap(diff)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folds = {f: build(args.root, f) for f in ('f0', 'f1', 'f2')}
    oof = []
    for held, test in folds.items():
        train = [r for f, rows in folds.items() if f != held for r in rows]
        assert not {r['qid'] for r in test} & {r['qid'] for r in train}
        oof.extend(predict(test, fit(train)))
    final = fit([r for rows in folds.values() for r in rows])
    dev = predict(build(args.root, 'dev'), final)
    assert len(oof) == 969 and len(dev) == 519
    out = {'design': 'Rank candidate documents outside the frozen ten, using the same dense top100. '
           'Train the supervised relevance prior on training folds only. Compare the first six and '
           'twenty challenger IDs to best rank union. No judge decision or reranking is executed. '
           'Reachable zero query labels are used to report subgroup coverage, never for routing.',
           'summary': {'oof': summarize(oof), 'dev': summarize(dev)},
           'per_query': {'oof': oof, 'dev': dev}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(out, stream, indent=1)
    print(json.dumps(out['summary'], indent=1))


if __name__ == '__main__':
    main()
