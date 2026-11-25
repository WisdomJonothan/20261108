"""Learn a one-document repair for the two-excerpt top ten."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_rerank_headroom_study import diagnose, rank
from replay_anchor_edit_scope import paired_delta, z
from replay_evidence_query_router import ARMS, cached_queries
from replay_anchor_fallback import index


DISCOUNT = 1 / np.log2(2 + np.arange(10))
POLICIES = {'keep_dispersed': (None, 0.)}
for source in ('restore', 'union'):
    for threshold in (0., .02, .05):
        POLICIES[f'{source}_{threshold:g}'] = (source, threshold)


def prepare(manifest, paths):
    out = []
    for row, scores in cached_queries(manifest, paths):
        docs = row['docs']
        temporal, dispersed = (rank(docs, scores[k], ARMS[k]) for k in ('temporal', 'dispersed'))
        values = {name: np.array([mapping[d] for d in docs]) for name, mapping in scores.items()}
        near, extra = (z(values[k]) for k in ('temporal', 'dispersed'))
        ref = np.array([DISCOUNT[i] if i < 10 else 0. for i in range(30)])
        x = np.column_stack((1 / np.log2(2 + np.arange(30)), ref, near, extra, np.minimum(near, extra),
                             np.abs(near - extra), np.tanh(values['temporal']/5),
                             np.tanh(values['dispersed']/5),
                             [doc in temporal for doc in docs], [doc in dispersed for doc in docs]))
        out.append({k: row[k] for k in ('domain', 'qid', 'split', 'cohort', 'actual_reference', 'docs')} |
                   {'original': docs[:10], 'dispersed': dispersed, 'temporal': temporal, 'features': x})
    return out


def label(rows, split, data):
    if {r['split'] for r in rows} != {split}:
        raise ValueError(f'Expected {split} rows')
    qrels = {d: read_qrels(data / d / f'qrels_{split}.txt') for d in {r['domain'] for r in rows}}
    return [r | {'gold': qrels[r['domain']][r['qid']]} for r in rows]


def fit(rows):
    counts = Counter(r['domain'] for r in rows)
    features = np.concatenate([r['features'] for r in rows])
    labels = [doc in r['gold'] for r in rows for doc in r['docs']]
    weights = np.repeat([len(rows) / (len(counts)*counts[r['domain']]) for r in rows],
                        [len(r['docs']) for r in rows])
    model = LogisticRegression(C=.2, solver='lbfgs', max_iter=600)
    model.fit(features, labels, sample_weight=weights)
    return model


def repair(row, prob, source, threshold):
    start = row['dispersed']
    pool = set(row['original'])
    if source == 'union':
        pool.update(row['temporal'])
    additions = [doc for doc in row['docs'] if doc in pool and doc not in start]
    if not additions:
        return start.copy()
    added = max(additions, key=prob.__getitem__)
    slot = max(range(1, 10), key=lambda i: DISCOUNT[i] * (prob[added] - prob[start[i]]))
    if DISCOUNT[slot] * (prob[added] - prob[start[slot]]) <= threshold:
        return start.copy()
    result = start.copy()
    result[slot] = added
    return result


def predict(rows, model):
    probs = model.predict_proba(np.concatenate([r['features'] for r in rows]))[:, 1].reshape(-1, 30)
    out = []
    for row, values in zip(rows, probs, strict=True):
        prob = dict(zip(row['docs'], values, strict=True))
        rankings = {'baseline': row['original'], 'keep_dispersed': row['dispersed']}
        rankings.update({name: repair(row, prob, source, threshold)
                         for name, (source, threshold) in POLICIES.items() if source})
        out.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'actual_reference')} |
                   {'rankings': rankings})
    return out


def grade(rows, labeled):
    gold = {(r['domain'], r['qid']): r['gold'] for r in labeled}
    return [r | {'scores': {name: ndcg10(rank, gold[r['domain'], r['qid']])
                            for name, rank in r['rankings'].items()}} for r in rows]


def choose(rows):
    return max(POLICIES, key=lambda a: macro((r['domain'], r['scores'][a]) for r in rows))


def crossfit(train):
    folds = sorted({r['cohort'] for r in train})
    choices = {}
    oof = []
    for held in folds:
        learn = [r for r in train if r['cohort'] != held]
        test = [r for r in train if r['cohort'] == held]
        inner = []
        for inner_held in (f for f in folds if f != held):
            inner_learn = [r for r in learn if r['cohort'] != inner_held]
            inner_test = [r for r in learn if r['cohort'] == inner_held]
            inner.extend(grade(predict(inner_test, fit(inner_learn)), inner_test))
        choices[held] = choose(inner)
        oof.extend(grade(predict(test, fit(learn)), test))
    return oof, choices


def summary(rows, choices):
    arms = {k: [k] * len(rows) for k in POLICIES}
    arms['selected'] = [choices[r['cohort']] for r in rows]
    results = {}
    for name, arms_used in arms.items():
        entries = list(zip(rows, arms_used, strict=True))
        variation = [(r['domain'], r['scores'][a] - r['scores']['keep_dispersed']) for r, a in entries]
        results[name] = {
            'score': macro((r['domain'], r['scores'][a]) for r, a in entries),
            'change': paired_delta(variation),
            'edited_queries': sum(r['rankings']['keep_dispersed'] != r['rankings'][a] for r, a in entries),
        }
    return results


def compose(rows, choices, router, split):
    routed = index({'per_query': router[f'{split}_per_query']})
    if {(r['domain'], r['qid']) for r in rows} != routed.keys():
        raise ValueError(f'Router and repair query populations differ for {split}')
    result = []
    for r in rows:
        route = routed[r['domain'], r['qid']]
        for name_old, name_new in (('baseline', 'baseline'), ('dispersed', 'keep_dispersed')):
            if route['rankings'][name_old] != r['rankings'][name_new]:
                raise ValueError(f'Router reference and repair rankings differ for {r["qid"]}')
        policy = router['selected'] if split == 'dev' else router['choices_by_fold'][r['cohort']]
        chosen = route['decisions'][policy]
        repair_arm = choices[r['cohort']]
        rankings = {'baseline': r['rankings']['keep_dispersed'], 'router': route['rankings'][chosen],
                    'repair': r['rankings'][repair_arm]}
        scores = {'baseline': r['scores']['keep_dispersed'], 'router': route['scores'][chosen],
                  'repair': r['scores'][repair_arm]}
        used = 'repair' if chosen == 'dispersed' else 'router'
        rankings['combined'], scores['combined'] = rankings[used], scores[used]
        result.append({'domain': r['domain'], 'qid': r['qid'], 'router_action': chosen,
                       'rankings': rankings, 'scores': scores})
    stats = {arm: {'score': macro((r['domain'], r['scores'][arm]) for r in result),
                   'against_dispersed': paired_delta([(r['domain'], r['scores'][arm]-r['scores']['baseline'])
                                                       for r in result])}
             for arm in ('router', 'repair', 'combined')}
    stats['combined']['against_repair'] = paired_delta([
        (r['domain'], r['scores']['combined']-r['scores']['repair']) for r in result
    ])
    return {'summary': stats, 'per_query': result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('training-manifest', 'dev-manifest', 'data-root', 'output'):
        ap.add_argument(f'--{name}', type=Path, required=True)
    for name in ('training-scores', 'dev-scores'):
        ap.add_argument(f'--{name}', type=Path, nargs='+', required=True)
    ap.add_argument('--router-report', type=Path)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a fresh output path')

    training = label(prepare(args.training_manifest, args.training_scores), 'train', args.data_root)
    oof, choices = crossfit(training)
    selected = choose(oof)
    print('Training selection', selected, choices, flush=True)
    model = fit(training)
    pending = prepare(args.dev_manifest, args.dev_scores)
    train_keys = {(r['domain'], r['qid']) for r in training}
    if len(pending) != 519 or any((r['domain'], r['qid']) in train_keys for r in pending):
        raise ValueError('Unexpected official/dev population')
    dev = grade(predict(pending, model), label(pending, 'dev', args.data_root))
    for r in dev:
        start = r['rankings']['keep_dispersed']
        if any(len(rank) != 10 or len(set(rank)) != 10 or rank[0] != start[0]
               or sum(d != e for d, e in zip(rank, start, strict=True)) > 1
               for name, rank in r['rankings'].items() if name in POLICIES):
            raise ValueError(f'Unexpected list changes for {r["qid"]}')
    report = {'selected': selected, 'choices': choices, 'policy': POLICIES,
              'training': summary(oof, choices),
              'training_actual_reference': summary([r for r in oof if r['actual_reference']], choices),
              'dev': summary(dev, {'dev': selected}),
              'dev_diagnostics': diagnose(dev, selected, args.data_root),
              'oof_per_query': oof, 'dev_per_query': dev}
    if abs(report['dev']['keep_dispersed']['score'] - .39488102030236294) > 1e-12:
        raise ValueError('Baseline score changed')
    if args.router_report:
        router = load(args.router_report)
        report['composition'] = {'oof': compose(oof, choices, router, 'oof'),
                                 'dev': compose(dev, {'dev': selected}, router, 'dev')}
        print('Composed', {s: {a: m['score'] for a, m in report['composition'][s]['summary'].items()}
                           for s in ('oof', 'dev')}, flush=True)
    print({k: (v['score'], v['edited_queries']) for k, v in report['dev'].items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
