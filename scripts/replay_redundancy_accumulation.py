"""Select a redundancy rule on disjoint fold queries and replay it with 8B."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from benchmark_pairwise_replacement_v2 import passages, strip_html, terms
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index
from replay_shortlist_near_duplicates import greedy, shingles, similarities

BASE = 'max_0.25'
MODES = {
    BASE: ('max', .25), 'max_0.5': ('max', .5), 'max_1': ('max', 1.),
    'sum_0.125': ('sum', .125), 'sum_0.25': ('sum', .25), 'sum_0.5': ('sum', .5),
    'sum_capped_0.25': ('capped', .25), 'sum_capped_0.5': ('capped', .5),
}


def rerank(docs, values, sim, mode, strength):
    if mode == 'max':
        return greedy(docs, values, sim, strength)
    remaining = values.copy()
    result = []
    running = np.zeros(len(docs))
    for _ in range(10):
        penalty = np.minimum(running, 2.) if mode == 'capped' else running
        i = int(np.argmax(remaining - strength * penalty))
        result.append(docs[i])
        remaining[i] = -np.inf
        running += sim[i]
    return result


def grade(domain, qid, docs, value, sim, gold):
    orders = {name: rerank(docs, value, sim, mode, strength)
              for name, (mode, strength) in MODES.items()}
    if any(len(set(d)) != 10 or d[0] != orders[BASE][0] for d in orders.values()):
        raise ValueError(f'Redundancy changed the first document for {qid}')
    return {'domain': domain, 'qid': qid, 'rankings': orders,
            'scores': {name: ndcg10(order, gold) for name, order in orders.items()}}


def proxy(args):
    root = args.root
    excluded = set().union(*(index(load(p)).keys() for p in args.exclude))
    six = index(load(args.reference))
    queries = {}
    for fold in ('f0', 'f1', 'f2'):
        for r in load(root / 'results/replacement_v2' / f'candidates_train_{fold}.json')['queries']:
            key = r['domain'], r['qid']
            if key in queries:
                raise ValueError(f'Query appears in more than one fold for {key}')
            queries[key] = r | {'fold': fold}
    by_domain = defaultdict(list)
    for (d, q), r in six.items():
        if (d, q) not in excluded and r['policies']['external6']['chosen'] is None:
            by_domain[d].append(queries[d, q])
    result = []
    for domain, rows in sorted(by_domain.items()):
        dense = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_train.trec')
        folded = {fold: read_trec_run(root / 'runs' / f'trainfuse_{fold}_{domain}_train.trec')
                  for fold in {r['fold'] for r in rows}}
        prepared = []
        for r in sorted(rows, key=lambda x: x['qid']):
            q = r['qid']
            start = r['frozen_top10']
            dense_docs = [doc for _, _, doc in dense[q][:100]]
            frozen = {doc: score for _, score, doc in folded[r['fold']][q][:100]}
            if set(frozen) != set(dense_docs) or list(frozen)[:10] != start:
                raise ValueError(f'Archived pools differ for {domain}/{q}')
            ids = set(start + [doc for doc in dense_docs if doc not in start][:24])
            docs = [doc for doc in dense_docs if doc in ids]
            if len(docs) != 34:
                raise ValueError(f'Incomplete proxy candidate set for {domain}/{q}')
            prepared.append((q, r['fold'], docs, .7 * z([frozen[doc] for doc in docs]), strip_html(r['query'])))
        required = {doc for _, _, docs, _, _ in prepared for doc in docs}
        corpus = {}
        with (args.data_root / domain / 'documents.jsonl').open(encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line)
                if doc['id'] in required:
                    corpus[doc['id']] = doc['content']
                    if corpus.keys() == required:
                        break
        if corpus.keys() != required:
            raise ValueError(f'Corpus incomplete for {domain}')
        gold = read_qrels(args.data_root / domain / 'qrels_train.txt')
        for q, fold, docs, value, text in prepared:
            vocabulary = terms(text)
            sim = similarities([shingles(passages(corpus[d], vocabulary)[0]) for d in docs])
            result.append(grade(domain, q, docs, value, sim, gold[q]) | {'fold': fold})
        print(domain, len(result), 'proxy queries', flush=True)
    return result


def crossfit_proxy(rows):
    choices = {}
    folds = {}
    for fold in sorted({r['fold'] for r in rows}):
        train = [r for r in rows if r['fold'] != fold]
        choices[fold] = max(MODES, key=lambda arm: macro((r['domain'], r['scores'][arm]) for r in train))
        folds[fold] = paired_delta([
            (r['domain'], r['scores'][choices[fold]] - r['scores'][BASE]) for r in rows if r['fold'] == fold
        ])
    paired = paired_delta([(r['domain'], r['scores'][choices[r['fold']]] - r['scores'][BASE]) for r in rows])
    return {'choices': choices, 'paired_by_fold': folds, 'paired': paired}


def current(args):
    archive = load(args.budgeted)
    old = index(load(args.duplicate))
    reports = {}
    choices = {}
    for label, paths, frozen_path in (('oof', args.oof, args.oof_frozen),
                                      ('dev', [args.dev], args.dev_frozen)):
        strong = {}
        for path in paths:
            strong.update(index(load(path)))
        full = index(load(frozen_path))
        records = [r for r in archive[f'{label}_per_query'] if r['active']]
        gold = {d: read_qrels(args.data_root / d / f'qrels_{"train" if label=="oof" else "dev"}.txt')
                for d in {r['domain'] for r in records}}
        result = []
        for r in records:
            d, q = r['domain'], r['qid']
            key = d, q
            snapshot = old[key]
            base = full[key]['frozen_scores']
            judge = strong[key]['scores']
            start = strong[key]['arms']['base']['ranking']
            reach = set(start + [doc for doc, _ in base if doc not in start][:24])
            docs = [doc for doc, _ in base if doc in reach]
            if len(docs) != 34:
                raise ValueError('Missing candidates')
            positions = {doc: i for i, doc in enumerate(docs)}
            sim = np.zeros((len(docs), len(docs)))
            for a, b, w in snapshot['near_duplicate_edges']:
                i, j = positions[a], positions[b]
                sim[i, j] = sim[j, i] = w
            values = .7 * z([s for doc, s in base if doc in reach]) + .3 * z([judge[doc] for doc in docs])
            item = grade(d, q, docs, values, sim, gold[d][q])
            if item['rankings'][BASE] != snapshot['rankings']['0.25']:
                raise ValueError(f'Reference duplicate policy differs for {key}')
            result.append(item | {'cohort': strong[key]['fold']})
        ranked = {(r['domain'], r['qid']): r for r in result}
        evaluated = {}
        duplicate = load(args.duplicate)
        baseline = (duplicate['oof']['0.25']['ndcg'] if label == 'oof'
                    else duplicate['dev']['selected']['ndcg'])
        for arm in MODES:
            change = paired_delta([(r['domain'], (ranked[r['domain'], r['qid']]['scores'][arm] -
                                                  ranked[r['domain'], r['qid']]['scores'][BASE]) if r['active'] else 0.)
                                   for r in archive[f'{label}_per_query']])
            evaluated[arm] = {'ndcg': baseline + change['point'], 'change': change}
            print(label, arm, evaluated[arm]['ndcg'], change['ci95'], flush=True)
        reports[label] = {'arms': evaluated, 'per_query': result}
    proxy_report = load(args.proxy_report)
    proxy_keys = {(r['domain'], r['qid']) for r in proxy_report['per_query']}
    actual_keys = {(r['domain'], r['qid']) for label in ('oof', 'dev') for r in archive[f'{label}_per_query']}
    if not proxy_keys.isdisjoint(actual_keys):
        raise ValueError('Proxy selection data must be disjoint from scored evaluation queries')
    reports['proxy_crossfit'] = crossfit_proxy(proxy_report['per_query'])
    choices['proxy'] = proxy_report['selected']
    choices['original_65'] = max(MODES, key=lambda arm: reports['oof']['arms'][arm]['ndcg'])
    reports['choices'] = choices
    print('Chosen by independent proxy and previous holdouts:', choices, flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--proxy', action='store_true')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--exclude', type=Path, nargs=2)
    parser.add_argument('--oof', type=Path, nargs=2)
    for name in ('proxy-report', 'dev', 'budgeted', 'duplicate', 'oof-frozen', 'dev-frozen'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.proxy:
        rows = proxy(args)
        scores = {arm: macro((r['domain'], r['scores'][arm]) for r in rows) for arm in MODES}
        selected = max(MODES, key=scores.get)
        report = {'queries': len(rows), 'domains': len({r['domain'] for r in rows}),
                  'selected': selected, 'scores': scores, 'per_query': rows}
        print('Proxy scores', scores, 'chosen rule', selected, flush=True)
    else:
        report = current(args)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, separators=(',', ':'))
        f.write('\n')


if __name__ == '__main__':
    main()
