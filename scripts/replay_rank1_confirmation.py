"""Replay fixed gates for moving a new document to rank one."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_accepted_weight_changes import orders, protect
from audit_anchor_dev import macro
from coverage_cf_bound import ndcg10, read_qrels
from compare_anchor_reference import load
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from replay_anchor_fusion_calibration import build, rank, score, weight

FREE = ('protect', 'frozen_top3', 'judge_top1', 'both_original_scores')
PAID = ('protect', 'verify_pair_4b', 'verify_list_4b')
ALL = FREE + PAID[1:] + ('unrestricted', 'oracle')


def margin_features(docs, frozen, judge, before, after):
    incumbent, challenger = before[0], after[0]
    i, j = docs.index(incumbent), docs.index(challenger)
    return [float(frozen[j] - frozen[i]), float(judge[j] - judge[i]),
            float(after.index(incumbent) + 1)]


def replay(rows, scores4b, scores8b):
    results = []
    for row in rows:
        item = {k: row[k] for k in ('domain', 'qid', 'cohort', 'route')}
        if row['route'] == 'fallback':
            results.append(item | {'deltas': dict.fromkeys(ALL, 0.), 'decisions': dict.fromkeys(ALL, False)})
            continue
        before, after = orders(row)
        base = protect(before, after, 1)
        incumbent, candidate = before[0], after[0]
        different = incumbent != candidate
        key = row['domain'], row['qid']
        smaller, strong = scores4b[key], scores8b[key]
        small = smaller.get('judge_scores', smaller.get('scores', {}))
        if (smaller['anchor'] != strong['anchor'] or
                not set(row['docs']) <= small.keys()):
            raise ValueError(f'Incomplete matched judge scores for {key}')
        docs = row['docs']
        features = row['features']
        frozen = {docs[i] for i in np.argsort(-features[:, 0], kind='stable')[:3]}
        judge_head = docs[int(np.argmax(features[:, 1]))]
        features4b = features.copy()
        features4b[:, 1] = z(np.array([small[d] for d in docs]))
        small_head = rank(row | {'features': features4b}, weight(.6))[0]
        change = score(row, after) - score(row, base)
        conditions = {
            'protect': False,
            'frozen_top3': candidate in frozen,
            'judge_top1': candidate == judge_head,
            'both_original_scores': candidate in frozen and candidate == judge_head,
            'verify_pair_4b': small[candidate] > small[incumbent],
            'verify_list_4b': candidate == small_head,
            'unrestricted': True,
            'oracle': change > 0.,
        }
        decisions = {name: different and allowed for name, allowed in conditions.items()}
        results.append(item | {
            'rank1_changed_by_unrestricted': different,
            'incumbent': incumbent, 'candidate': candidate, 'reference': strong['anchor'],
            'margin_features': margin_features(docs, features[:, 0], features[:, 1], before, after),
            'protected_ranking': base, 'unrestricted_ranking': after,
            'rankings_ndcg': {'protect': score(row, base), 'unrestricted': score(row, after)},
            'decisions': decisions,
            'deltas': {name: change if swap else 0. for name, swap in decisions.items()},
        })
    return results


def select(rows, options):
    return max(options, key=lambda name: macro((r['domain'], r['deltas'][name]) for r in rows))


def compare(rows, baseline, arms):
    summary = {}
    for label, chooser in arms.items():
        picks = [(r, chooser(r)) for r in rows]
        effect = paired_delta([(r['domain'], r['deltas'][arm]) for r, arm in picks])
        counts = Counter()
        prompts = 0
        for r, arm in picks:
            if r['decisions'][arm]:
                value = r['deltas'][arm]
                counts['helpful' if value > 1e-12 else 'harmful' if value < -1e-12 else 'neutral'] += 1
            if r.get('rank1_changed_by_unrestricted'):
                anchor = r['reference']
                if arm == 'verify_pair_4b':
                    prompts += 2 * sum(r[name] != anchor for name in ('incumbent', 'candidate'))
                if arm == 'verify_list_4b':
                    prompts += 2 * (10 - int(anchor in r['protected_ranking']))
        summary[label] = {
            'ndcg': baseline + effect['point'], 'change': effect,
            'rank1_changes': dict(counts), 'extra_4b_prompts': prompts,
            'extra_4b_prompts_per_query': prompts / len(rows),
        }
    return summary


def load_scores(paths):
    combined = {}
    for path in paths:
        part = index(load(path))
        if not part.keys().isdisjoint(combined):
            raise ValueError('Overlapping score archives')
        combined.update(part)
    return combined


def fit_margin(rows):
    x = np.array([r['features'] for r in rows])
    y = np.array([r['delta'] for r in rows])
    counts = Counter(r['domain'] for r in rows)
    w = np.array([1. / (len(counts) * counts[r['domain']]) for r in rows])
    center = w @ x
    scales = np.sqrt(w @ ((x - center) ** 2)).clip(min=1e-8)
    xn = (x - center) / scales
    intercept = float(w @ y)
    coefficients = np.linalg.solve(xn.T @ (w[:, None] * xn) + .1 * np.eye(x.shape[1]),
                                   xn.T @ (w * (y - intercept)))
    return center, scales, intercept, coefficients


def estimate(row, model):
    center, scales, intercept, coefficients = model
    return float(intercept + ((np.array(row['features']) - center) / scales) @ coefficients)


def summarize_margin(rows, baseline):
    result = {}
    for name, threshold in (('positive', 0.), ('above_0.01', .01), ('protect', float('inf'))):
        actions = [r['prediction'] > threshold for r in rows]
        paired = paired_delta([(r['domain'], r['delta'] if take else 0.) for take, r in zip(actions, rows)])
        result[name] = {'score': baseline + paired['point'], 'change': paired,
                        'changes_selected': sum(actions)}
    return result


def transfer(source, data, oof, dev, baseline):
    rows = source['per_query']
    gold = {d: read_qrels(data / d / 'qrels_train.txt') for d in {r['domain'] for r in rows}}
    training = []
    for r in rows:
        docs = r['documents']
        before, after = (r['rankings'][name] for name in ('0.3', '0.6'))
        different = before[0] != after[0]
        protected = protect(before, after, 1)
        labels = gold[r['domain']][r['qid']]
        training.append({
            'domain': r['domain'], 'qid': r['qid'], 'eligible': different,
            'features': margin_features(docs, np.array([r['frozen_z'][d] for d in docs]),
                                        z(np.array([r['judge_scores'][d] for d in docs])), before, after),
            'delta': ndcg10(after, labels) - ndcg10(protected, labels),
            'protected_ndcg': ndcg10(protected, labels),
        })
    eligible = [r for r in training if r['eligible']]
    for domain in gold:
        model = fit_margin([r for r in eligible if r['domain'] != domain])
        for r in training:
            if r['domain'] == domain:
                r['prediction'] = estimate(r, model) if r['eligible'] else float('-inf')
    protected = macro((r['domain'], r['protected_ndcg']) for r in training)
    crossfit = summarize_margin(training, protected)
    chosen = max(('protect', 'above_0.01', 'positive'), key=lambda arm: crossfit[arm]['score'])
    model = fit_margin(eligible)
    reports = {'training_crossfit': crossfit, 'chosen_threshold': chosen,
               'training_candidates': len(eligible), 'fit_parameters': [np.asarray(x).tolist() for x in model]}
    for label, archive, split in (('oof_8b', oof, 'oof'), ('dev_8b', dev, 'dev')):
        target = []
        for r in archive:
            row = {'domain': r['domain'], 'delta': r['deltas']['unrestricted']}
            row['prediction'] = estimate({'features': r['margin_features']}, model) if r.get(
                'rank1_changed_by_unrestricted'
            ) else float('-inf')
            r['margin_estimate'] = row['prediction'] if r.get('rank1_changed_by_unrestricted') else None
            target.append(row)
        reports[label] = summarize_margin(target, baseline[split])
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgeted', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--oof-8b', type=Path, nargs=2, required=True)
    parser.add_argument('--oof-4b', type=Path, nargs=2, required=True)
    parser.add_argument('--dev-8b', type=Path, required=True)
    parser.add_argument('--dev-4b', type=Path, required=True)
    parser.add_argument('--oof-frozen', type=Path, required=True)
    parser.add_argument('--dev-frozen', type=Path, required=True)
    parser.add_argument('--train-4b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archive = load(args.budgeted)
    previous = load(args.previous)
    oof = build(archive, args.oof_8b, args.oof_frozen, args.data_root, 'oof')
    dev = build(archive, [args.dev_8b], args.dev_frozen, args.data_root, 'dev')
    oof = replay(oof, load_scores(args.oof_4b), load_scores(args.oof_8b))
    dev = replay(dev, load_scores([args.dev_4b]), load_scores([args.dev_8b]))
    result = {
        'protocol': 'Compare keeping the existing first result against releasing the first position to '
                    'the 60-percent 8B reorder. Select low-cost and 4B-assisted gates on training. '
                    'Fit a separate margin model on 180 disjoint 4B training queries.',
        'oof_per_query': oof, 'dev_per_query': dev, 'selected': {}, 'oof': {}, 'dev': {},
    }
    controls = {arm: (lambda r, selected=arm: selected) for arm in ALL}
    base = {split: previous[split]['full_policy']['1']['ndcg'] for split in ('oof', 'dev')}
    for label, options in (('free', FREE), ('4b', PAID)):
        pooled = select(oof, options)
        crossfit = {i: select([r for r in oof if r['cohort'] != i], options) for i in (0, 1)}
        result['selected'][label] = {'pooled': pooled, 'crossfit': crossfit}
        result['oof'].update(compare(oof, base['oof'], {f'selected_{label}': lambda r: crossfit[r['cohort']]}))
        result['dev'].update(compare(dev, base['dev'], {f'selected_{label}': lambda r: pooled}))
    for split, rows in (('oof', oof), ('dev', dev)):
        result[split].update(compare(rows, base[split], controls))
        for arm, item in result[split].items():
            print(split, arm, round(item['ndcg'], 7), item['change']['ci95'],
                  item['rank1_changes'], item['extra_4b_prompts'])
    print('Selected on training:', result['selected'])
    result['margin_transfer'] = transfer(load(args.train_4b), args.data_root, oof, dev, base)
    print('Margin transfer threshold:', result['margin_transfer']['chosen_threshold'])
    for group in ('training_crossfit', 'oof_8b', 'dev_8b'):
        for arm, item in result['margin_transfer'][group].items():
            print(group, arm, item['score'], item['change']['point'], item['change']['ci95'],
                  item['changes_selected'])
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
