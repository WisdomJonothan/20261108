"""Learn a cross-retriever reranker on fold predictions from the training set."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from eval_rerank_headroom_study import summary


POLICIES = {
    'same_ten': (10, .30), 'same_ten_strong': (10, .55),
    'dense100': (100, .30), 'dense100_strong': (100, .55),
    'union124': (124, .30), 'union124_strong': (124, .55),
    'ranker_dense': (100, 1.), 'ranker_union': (124, 1.),
}


def reference(root, split):
    analysis = root / 'results' / 'analysis'
    exact = load(analysis / 'directional_passage_containment_20261125.json')['per_query']
    protected = load(analysis / 'accepted_weight_full_dev_audit_20261125.json')
    group = 'dev' if split == 'dev' else 'oof'
    orders = {(r['domain'], r['qid']): r['rankings']['double90'] for r in exact if r['sample'] == group}
    orders.update({(r['domain'], r['qid']): r['rankings']['1'] for r in protected[group]['per_query']})
    if len(orders) != (519 if split == 'dev' else 65):
        raise ValueError('Reference membership is inconsistent')
    return orders


def features(docs, run, scaling_depth):
    values = np.array([score for _, score, _ in run[:scaling_depth]])
    mean, std = values.mean(), max(float(values.std()), 1e-8)
    ranks = {doc: (rank, score) for rank, score, doc in run}
    signals = []
    for doc in docs:
        rank, score = ranks.get(doc, (len(run) + 1, values[-1]))
        signals.append(((score - mean) / std, 1 / np.log2(1 + rank), float(doc in ranks)))
    return np.array(signals)


def build(root, data, runs, bge_dir, split):
    frozen_start = reference(root, split)
    folds = ('dev',) if split == 'dev' else ('f0', 'f1', 'f2')
    entries = []
    for fold in folds:
        name = 'candidates_dev' if split == 'dev' else f'candidates_train_{fold}'
        entries.extend((fold, r) for r in load(root / 'results' / 'replacement_v2' / f'{name}.json')['queries'])
    actions = ({} if split == 'dev' else
               index(load(root / 'results' / 'analysis' / 'position_oracle_oof_20261124.json')))
    rows = []
    for domain in sorted({r['domain'] for _, r in entries}):
        dense = read_trec_run(runs / f'dense_mxbai_{domain}_{split}.trec')
        bge = read_trec_run(bge_dir / f'dense_bge_{domain}_{split}.trec')
        merged = {}
        for fold in folds:
            prefix = 'sub_sft_e2bc' if fold == 'dev' else f'trainfuse_{fold}'
            merged[fold] = read_trec_run(runs / f'{prefix}_{domain}_{split}.trec')
        qrels = read_qrels(data / domain / f'qrels_{split}.txt')
        for fold, r in ((f, r) for f, r in entries if r['domain'] == domain):
            qid = r['qid']
            key = domain, qid
            canonical = [doc for _, _, doc in dense[qid][:100]]
            starting = frozen_start.get(key)
            if starting is None:
                starting = r['frozen_top10'].copy()
                proposal = actions[key]['policies']['external6']['chosen']
                if proposal is not None:
                    slot = proposal['slot'] - 1
                    if starting[slot] != proposal['incumbent']:
                        raise ValueError(f'Six-pair ranking differs for {key}')
                    starting[slot] = proposal['challenger']
            cset = set(canonical)
            extra = [doc for _, _, doc in bge[qid][:300] if doc not in cset][:24]
            if len(extra) != 24 or len(cset) != 100 or len(set(starting)) != 10 or not set(starting) <= cset:
                raise ValueError(f'Unexpected scoring pool for {key}')
            docs = starting + [doc for doc in canonical if doc not in starting] + extra
            run = merged[fold][qid][:100]
            if {doc for _, _, doc in run} != cset:
                raise ValueError(f'First-stage fusion differs for {key}')
            mx = features(docs, dense[qid][:300], 100)
            alt = features(docs, bge[qid][:300], 100)
            old = features(docs, run, 100)
            x = np.column_stack((mx, alt, old, np.minimum(mx[:, 1], alt[:, 1]),
                                 mx[:, 0] - old[:, 0]))
            rows.append({'domain': domain, 'qid': qid, 'cohort': fold, 'docs': docs,
                         'features': x, 'gold': qrels[qid]})
        print(split, domain, 'ready', flush=True)
    unique = {(r['domain'], r['qid']) for r in rows}
    if len(rows) != len(unique) or len(rows) != (519 if split == 'dev' else 969):
        raise ValueError(f'Unexpected population for {split}')
    return rows


def fit(rows):
    counts = Counter(r['domain'] for r in rows)
    x = np.concatenate([r['features'] for r in rows])
    y = np.array([doc in r['gold'] for r in rows for doc in r['docs']])
    weights = np.concatenate([np.full(len(r['docs']), len(rows) / (len(counts) * counts[r['domain']]))
                              for r in rows])
    diverse = {
        'linear': LogisticRegression(C=.1, max_iter=500),
        'trees': HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=9,
                                                min_samples_leaf=100, l2_regularization=10.,
                                                early_stopping=False, random_state=20261125),
    }
    for model in diverse.values():
        model.fit(x, y, sample_weight=weights)
    return diverse


def predict(rows, models):
    result = []
    x = np.concatenate([r['features'] for r in rows])
    signals = {name: model.decision_function(x) for name, model in models.items()}
    cursor = 0
    for r in rows:
        docs, gold = r['docs'], r['gold']
        data = {name: scores[cursor:cursor+len(docs)] for name, scores in signals.items()}
        cursor += len(docs)
        arms = {'baseline': docs[:10]}
        for name, prediction in data.items():
            for policy, (width, weight) in POLICIES.items():
                prior = z([1 / np.log2(2 + i) for i in range(10)] +
                          [1 / np.log2(12)] * (width - 10))
                blend = (1 - weight) * prior + weight * z(prediction[:width])
                ranking = np.argsort(-blend, kind='stable')[:10]
                arms[f'{name}/{policy}'] = [docs[i] for i in ranking]
        result.append({k: r[k] for k in ('domain', 'qid', 'cohort')} |
                      {'rankings': arms, 'scores': {arm: ndcg10(rank, gold) for arm, rank in arms.items()},
                       'oracle_dense': ndcg10(sorted(set(docs[:100]) & gold), gold),
                       'oracle_union': ndcg10(sorted(set(docs) & gold), gold)})
    return result


def choose(rows):
    return max(rows[0]['scores'], key=lambda a: macro((r['domain'], r['scores'][a]) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    for name in ('data-root', 'runs-root', 'bge-train-root', 'bge-dev-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')

    with threadpool_limits(limits=4):
        train = build(args.root, args.data_root, args.runs_root, args.bge_train_root, 'train')
        oof, choices = [], {}
        for fold in ('f0', 'f1', 'f2'):
            fit_rows = [r for r in train if r['cohort'] != fold]
            inner = []
            for held in sorted({r['cohort'] for r in fit_rows}):
                inner_model = fit([r for r in fit_rows if r['cohort'] != held])
                inner.extend(predict([r for r in fit_rows if r['cohort'] == held], inner_model))
            choices[fold] = choose(inner)
            model = fit(fit_rows)
            oof.extend(predict([r for r in train if r['cohort'] == fold], model))
        fitted = fit(train)
        choice = choose(oof)
        print('Training choices', choice, choices, flush=True)
        dev_rows = build(args.root, args.data_root, args.runs_root, args.bge_dev_root, 'dev')
        dev = predict(dev_rows, fitted)
    result = {'oof_choice': choices, 'pooled_choice': choice, 'policy': POLICIES,
              'oof': summary(oof, choices), 'dev': summary(dev, {'dev': choice}),
              'oof_per_query': oof, 'dev_per_query': dev}
    for split, rows in (('oof', oof), ('dev', dev)):
        result[split]['union_oracle'] = paired_delta(
            [(r['domain'], r['oracle_union'] - r['oracle_dense']) for r in rows])
        print(split, {a: v['score'] for a, v in result[split].items() if 'score' in v}, flush=True)
    if abs(result['dev']['baseline']['score'] - .39269165354596675) > 1e-12:
        raise ValueError('Official baseline changed')
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
