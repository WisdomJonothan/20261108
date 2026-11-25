"""Cross-fit a query-level ranking selector on cached evidence scores."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_rerank_headroom_study import diagnose, rank, registers
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index


ARMS = {'baseline': 'reference', 'temporal': 'near_conservative', 'dispersed': 'deeper_head_fixed'}
FEATURES = (
    'two_excerpts', 'admitted_fraction', 'same_position_fraction',
    'view_agreement', 'own_admission_margin', 'other_admission_margin',
    'own_discount_gain', 'other_discount_gain', 'first_document_score',
    'affirmative_displacement_balance',
)
SETTINGS = {'temporal': (None, 0.), 'dispersed': (None, 0.), 'baseline': (None, 0.)}
for shrinkage in (10., 40.):
    for gap in (0., .005, .015):
        SETTINGS[f'ridge{shrinkage:g}_gap{gap:g}'] = shrinkage, gap
    SETTINGS[f'ridge{shrinkage:g}_default_dispersed'] = shrinkage, .005


def features(docs, baseline, candidate, own, other, consensus, mode):
    admitted = [d for d in candidate if d not in baseline]
    removed = [d for d in baseline if d not in candidate]
    own_z, other_z = (dict(zip(docs, z([s[d] for d in docs]))) for s in (own, other))
    discount = 1 / np.log2(2 + np.arange(10))

    def lift(scores, discounted):
        before, after = (baseline, candidate) if discounted else (removed, admitted)
        if not before:
            return 0.
        weights = discount if discounted else np.ones(len(before)) / len(before)
        return float(weights @ np.array([scores[a] - scores[b] for a, b in zip(after, before, strict=True)]))

    return [
        float(mode == 'dispersed'), len(admitted)/10, sum(a == b for a, b in zip(baseline, candidate))/10,
        len(set(candidate) & set(consensus))/10, lift(own_z, False), lift(other_z, False),
        lift(own_z, True), lift(other_z, True), own_z[baseline[0]],
        (sum(own[d] > 0 for d in admitted) - sum(own[d] > 0 for d in removed))/10,
    ]


def cached_queries(manifest, source_paths):
    queries = index({'per_query': load(manifest)['queries']})
    sources = registers(source_paths)
    if set(sources) != {'temporal', 'dispersed'} or any(
        s['scores'].keys() != queries.keys() for s in sources.values()
    ):
        raise ValueError('Both evidence modes must cover the same queries')
    for (domain, qid), row in queries.items():
        docs = row['docs']
        scores = {name: s['scores'][domain, qid]['scores'] for name, s in sources.items()}
        if len(docs) != len(set(docs)) or len(docs) != 30 or any(s.keys() != set(docs) for s in scores.values()):
            raise ValueError(f'Candidate set differs for {domain}/{qid}')
        yield row, scores


def prepare(manifest, source_paths):
    records = []
    for row, scores in cached_queries(manifest, source_paths):
        docs = row['docs']
        orders = {'baseline': docs[:10]}
        orders.update({name: rank(docs, scores[name], ARMS[name]) for name in scores})
        tensor = [features(docs, orders['baseline'], orders[name], scores[name], scores[other], orders[other], name)
                  for name, other in (('temporal', 'dispersed'), ('dispersed', 'temporal'))]
        records.append({k: row[k] for k in ('domain', 'qid', 'split', 'cohort', 'actual_reference')} |
                       {'rankings': orders, 'features': tensor})
    return records


def grade(rows, data, split):
    if {r['split'] for r in rows} != {split}:
        raise ValueError(f'Expected only {split} rows')
    labels = {d: read_qrels(data / d / f'qrels_{split}.txt') for d in {r['domain'] for r in rows}}
    return [r | {'scores': {k: ndcg10(order, labels[r['domain']][r['qid']])
                            for k, order in r['rankings'].items()}} for r in rows]


def fit(rows, shrinkage):
    x = np.asarray([r['features'] for r in rows]).reshape(-1, len(FEATURES))
    target = np.asarray([r['scores'][name] - r['scores']['baseline']
                         for r in rows for name in ('temporal', 'dispersed')])
    per_domain = Counter(r['domain'] for r in rows)
    weights = np.repeat([len(rows) / (len(per_domain)*per_domain[r['domain']]) for r in rows], 2)
    mean = np.average(x, axis=0, weights=weights)
    std = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights)).clip(.00001)
    scaled = (x - mean) / std
    bias = np.average(target, weights=weights)
    coef = np.linalg.solve(scaled.T @ (weights[:, None] * scaled) + shrinkage * np.eye(len(FEATURES)),
                           scaled.T @ (weights * (target-bias)))
    return {'mean': mean.tolist(), 'std': std.tolist(), 'bias': float(bias), 'coef': coef.tolist()}


def predict(rows, trained):
    arms = {}
    for name, (shrinkage, gap) in SETTINGS.items():
        if shrinkage is None:
            arms[name] = [name] * len(rows)
            continue
        model = trained[shrinkage]
        xs = np.asarray([r['features'] for r in rows])
        p = ((xs - model['mean']) / model['std']) @ model['coef'] + model['bias']
        if name.endswith('default_dispersed'):
            arms[name] = ['temporal' if t - d > gap else 'dispersed' for t, d in p]
        else:
            arms[name] = ['temporal' if t >= d and t > gap else 'dispersed' if d > gap else 'baseline'
                          for t, d in p]
    return arms


def fit_predict(train, predicted):
    models = {s: fit(train, s) for s in (10., 40.)}
    return predict(predicted, models)


def assemble(rows, predictions):
    return [r | {'decisions': {name: choice[i] for name, choice in predictions.items()}}
            for i, r in enumerate(rows)]


def score(rows, policy):
    return macro((r['domain'], r['scores'][r['decisions'][policy]]) for r in rows)


def choose(rows):
    preference = ('dispersed', 'temporal', 'baseline', *(k for k in SETTINGS if k not in ARMS))
    return max(preference, key=lambda policy: score(rows, policy))


def crossfit(rows):
    fold_ids = sorted({r['cohort'] for r in rows})
    outer, choices = [], {}
    for held in fold_ids:
        train = [r for r in rows if r['cohort'] != held]
        validation = [r for r in rows if r['cohort'] == held]
        inner = []
        for fold in fold_ids:
            if fold == held:
                continue
            early = [r for r in train if r['cohort'] != fold]
            late = [r for r in train if r['cohort'] == fold]
            inner.extend(assemble(late, fit_predict(early, late)))
        choices[held] = choose(inner)
        outer.extend(assemble(validation, fit_predict(train, validation)))
    return outer, choices


def summarize(rows, decisions):
    differences = [(r['domain'], r['scores'][a] - r['scores']['dispersed'])
                   for r, a in zip(rows, decisions, strict=True)]
    paired = paired_delta(differences)
    domains = paired['per_domain']
    leave_one_out = {d: (sum(domains.values()) - value) / (len(domains) - 1)
                     for d, value in domains.items()}
    return {'score': macro((r['domain'], r['scores'][a]) for r, a in zip(rows, decisions, strict=True)),
            'versus_dispersed': paired, 'decisions': dict(Counter(decisions)),
            'leave_one_domain_out': leave_one_out}


def report(rows, selected, by_fold=None, controls=True):
    summary = {p: summarize(rows, [r['decisions'][p] for r in rows]) for p in SETTINGS} if controls else {}
    actions = [r['decisions'][by_fold[r['cohort']] if by_fold else selected] for r in rows]
    summary['selected'] = summarize(rows, actions)
    summary['oracle'] = {
        'score': macro((r['domain'], max(r['scores'].values())) for r in rows),
        'queries_above_dispersed': sum(max(r['scores'].values()) > r['scores']['dispersed'] + 1e-12 for r in rows),
    }
    return summary, actions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ('training-manifest', 'dev-manifest', 'data-root', 'output'):
        ap.add_argument(f'--{n}', type=Path, required=True)
    for n in ('training-scores', 'dev-scores'):
        ap.add_argument(f'--{n}', type=Path, required=True, nargs='+')
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Use a fresh output path')

    train = grade(prepare(args.training_manifest, args.training_scores), args.data_root, 'train')
    oof, choices = crossfit(train)
    selected = choose(oof)
    oof_summary, _ = report(oof, selected, choices)
    held65, _ = report([r for r in oof if r['actual_reference']], selected, choices)
    print('Training selection', selected, choices, oof_summary['selected'], flush=True)
    models = {s: fit(train, s) for s in (10., 40.)}
    pending_dev = prepare(args.dev_manifest, args.dev_scores)
    train_keys = {(r['domain'], r['qid']) for r in train}
    if len(pending_dev) != 519 or any((r['domain'], r['qid']) in train_keys for r in pending_dev):
        raise ValueError('Unexpected official/dev population')
    decisions = predict(pending_dev, models)
    dev = assemble(grade(pending_dev, args.data_root, 'dev'), decisions)
    dev_summary, actions = report(dev, selected)
    if abs(dev_summary['dispersed']['score'] - .39488102030236294) > 1e-12:
        raise ValueError('Full-dev two-excerpt reference changed')
    for r, action in zip(dev, actions, strict=True):
        r['rankings']['selected'] = r['rankings'][action]
    diagnostics = diagnose(dev, 'selected', args.data_root)
    final = {'selected': selected, 'choices_by_fold': choices, 'training': oof_summary,
             'training_actual_reference': held65, 'dev': dev_summary, 'diagnostics': diagnostics,
             'features': FEATURES, 'settings': SETTINGS, 'models': models,
             'oof_per_query': oof, 'dev_per_query': dev}
    print('Dev', {k: round(v['score'], 6) for k, v in dev_summary.items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(final, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
