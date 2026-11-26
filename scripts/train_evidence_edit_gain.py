"""Estimate the gain of a single edit to a fixed evidence ranking."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression

from compare_anchor_reference import load
from coverage_cf_bound import read_qrels
from eval_qwen_reranker_adapter import candidate_judges, grade, metric, summarize
from replay_anchor_cap import z
from replay_anchor_fallback import index


ARMS = ('current', 'pointwise', 'pointwise_guard', 'pair', 'pair_guard')
DISCOUNT = 1 / np.log2(2 + np.arange(10))
VOTERS = ('current', 'pointwise', 'reasoned', 'verified')


def combine(paths, field):
    result = index({'per_query': [row for path in paths for row in load(Path(path))['per_query']]})
    return {key: row[field] for key, row in result.items()}


def extract(path):
    inputs = load(path)
    heads = index(load(Path(inputs['head'])))
    voters = index(load(Path(inputs['consensus'])))
    categories = combine(inputs['categories'], 'categories')
    reasoning = combine(inputs['reasoning'], 'observations')
    judges = [candidate_judges([Path(x) for x in inputs[name]])[0]['pretrained']
              for name in ('first', 'second')]
    if not all(x.keys() == heads.keys() for x in (voters, categories, reasoning, *judges)):
        raise ValueError('Every score source must include the same question IDs')
    rows = []
    for key, row in heads.items():
        start = row['rankings']['verified_0.5']
        category, obs = categories[key], reasoning[key]
        documents = start + [d for d in category if d not in start]
        if (len(category) != 22 or len(obs) != 14 or not obs.keys() <= category.keys()
                or not all(set(documents) <= source[key]['scores'].keys() for source in judges)):
            raise ValueError(f'Document evidence missing for {key}')
        maps = [{d: DISCOUNT[i] for i, d in enumerate(voters[key]['rankings'][name])}
                for name in VOTERS]
        common = []
        for source in judges:
            logits = np.array([source[key]['scores'][d] for d in documents])
            common.extend((z(logits), np.tanh(logits/5)))
        matrix = []
        for d in documents:
            values = category.get(d)
            evidence = obs.get(d)
            odds = values[3] - np.logaddexp.reduce(values[:3]) if values else 0.
            conflict = values[2]-max(values[i] for i in (0,1,3)) if values else 0.
            matrix.append([*(m.get(d, 0.) for m in maps), values is not None, np.tanh(odds/5.),
                           np.tanh(conflict/5.), evidence is not None,
                           evidence['direct']/3 if evidence else 0.,
                           evidence['reasoned']/3 if evidence else 0.,
                           evidence['verified'] if evidence else 0.])
        x = np.column_stack((*common, np.asarray(matrix)))
        rows.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                    {'documents': documents, 'features': x})
    return rows


def moves(row):
    size = len(row['documents'])
    # Preserve the head for admissions. Within-list swaps may improve the head
    # without sacrificing membership.
    left = [i for i in range(1, 10) for _ in range(10, size)]
    right = list(range(10, size))*9
    for i in range(5):
        left.extend([i]*(9-i))
        right.extend(range(i+1, 10))
    i, j = np.array(left), np.array(right)
    return i, j, DISCOUNT[i] - np.r_[DISCOUNT, np.zeros(size-10)][j]


def pair_features(x, i, j):
    difference = x[j]-x[i]
    return np.column_stack((difference, abs(difference), (x[j]+x[i])/2))


def train(rows, data):
    count = Counter(r['domain'] for r in rows)
    gold = {d: read_qrels(data/d/'qrels_train.txt') for d in count}
    documents, labels, sample, pairs, changes, weight = [], [], [], [], [], []
    for r in rows:
        docs, x = r['documents'], r['features']
        relevant = np.array([d in gold[r['domain']][r['qid']] for d in docs], dtype=int)
        balance = 1/count[r['domain']]
        documents.append(x)
        labels.extend(relevant)
        sample.extend([balance/len(docs)]*len(docs))
        i, j, discount = moves(r)
        labels_pair = relevant[j]-relevant[i]
        pairs.extend((pair_features(x, i, j), pair_features(x, j, i)))
        changes.extend((labels_pair, -labels_pair))
        w = balance*discount/(2*discount.sum())
        weight.extend((w,w))
    point = LogisticRegression(C=.2, max_iter=500)
    sample = np.asarray(sample)
    point.fit(np.concatenate(documents), labels, sample_weight=sample/sample.mean())
    model = LogisticRegression(C=.2, max_iter=500)
    w = np.concatenate(weight)
    model.fit(np.concatenate(pairs), np.concatenate(changes), sample_weight=w/w.mean())
    if list(model.classes_) != [-1,0,1]:
        raise ValueError('Training needs harmful, neutral and beneficial edits')
    return point, model


def edit(docs, i, j):
    revised = docs[:10]
    revised[i] = docs[j]
    if j < 10:
        revised[j] = docs[i]
    return revised


def predict(rows, scores):
    point, model = scores
    result = []
    for r in rows:
        x, docs = r['features'], r['documents']
        i, j, disc = moves(r)
        p = point.predict_proba(x)[:, 1]
        f = model.predict_proba(pair_features(x, i, j))
        rvs = model.predict_proba(pair_features(x, j, i))
        pair_gain = (f[:, 2] - f[:, 0] + rvs[:, 0] - rvs[:, 2])/2
        orders = {'current': docs[:10]}
        for name, values in (('pointwise', (p[j]-p[i])*disc), ('pair', pair_gain*disc)):
            best = np.argmax(values)
            proposal = edit(docs, i[best], j[best])
            orders[name] = proposal if values[best] > 0 else docs[:10]
            orders[f'{name}_guard'] = proposal if values[best] > .02 else docs[:10]
        result.append({k: r[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                      {'rankings': orders})
    return result


def gains(rows):
    reference = metric(rows, ['current']*len(rows))
    return {arm: metric(rows, [arm]*len(rows))-reference for arm in ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, nargs=2, required=True)
    for name in ('data-root', 'output', 'dev', 'selection'):
        parser.add_argument(f'--{name}', type=Path, required=name in ('data-root', 'output'))
    args = parser.parse_args()
    if args.output.exists() or bool(args.dev) != bool(args.selection):
        parser.error('Use a fresh output and pair dev with its locked selection')
    first, second = (extract(p) for p in args.training)
    keys = [{(r['domain'],r['qid']) for r in rows} for rows in (first,second)]
    if list(map(len, keys)) != [186,155] or keys[0]&keys[1]:
        raise ValueError('Expected the two fixed disjoint training cohorts')

    if args.dev:
        summary_selection = load(args.selection)
        selected = summary_selection['selected']
        if selected not in ARMS:
            raise ValueError('Unknown locked decision')
        rows = grade(predict(extract(args.dev), train(first+second,args.data_root)), 'dev', args.data_root)
        summary = summarize(rows, selected)
        if len(rows) != 519 or abs(summary['current']['score']-.40674178096988606)>1e-12:
            raise ValueError('Expected the frozen 519-query reference')
        report = {'dev': summary}
    else:
        sides = [grade(predict(held, train(fit, args.data_root)), 'train', args.data_root)
                 for held, fit in ((first,second),(second,first))]
        rows = sides[0]+sides[1]
        results = [gains(side) for side in sides]
        joint = gains(rows)
        # Prioritize an effect that transfers in both directions.
        selected = max(ARMS,key=lambda arm: (min(s[arm] for s in results),joint[arm]))
        summary = summarize(rows, selected)
        report = {'training': summary, 'cohort_gains': results, 'queries_per_cohort': list(map(len,sides))}
    report.update({'selected': selected, 'per_query': rows})
    delta = summary['selected']['change']['per_domain']
    report['selected_minimum_leave_one_domain_out'] = min(
        mean(v for d,v in delta.items() if d != held) for held in delta)
    print(selected, {k: (round(v['score'],8),round(v['change']['point'],8)) for k,v in summary.items()},flush=True)
    if 'cohort_gains' in report:
        print('Transfer gains',report['cohort_gains'],flush=True)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(report,file,separators=(',',':'))


if __name__ == '__main__':
    main()
