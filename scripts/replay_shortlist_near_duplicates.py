"""Test near-duplicate penalties while keeping the 8B candidate budget fixed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

from audit_anchor_dev import macro
from benchmark_pairwise_replacement_v2 import passages, terms
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import paired_delta, z
from replay_anchor_fallback import index

PENALTIES = (0., .25, .5, 1.)


def shingles(text):
    words = re.findall(r'[a-z]+|[0-9]+', text.lower())
    return set(zip(words, words[1:], words[2:], words[3:], words[4:]))


def similarities(sets):
    result = np.zeros((len(sets), len(sets)))
    for i, a in enumerate(sets):
        for j in range(i):
            b = sets[j]
            if min(len(a), len(b)) < 12 or min(len(a), len(b)) / max(len(a), len(b)) < .8:
                continue
            intersection = len(a & b)
            similarity = intersection / (len(a) + len(b) - intersection)
            if similarity >= .8:
                result[i, j] = result[j, i] = similarity
    return result


def greedy(docs, values, sim, penalty):
    remain = values.copy()
    ranking = []
    maximum = np.zeros(len(docs))
    for _ in range(10):
        i = int(np.argmax(remain - penalty * maximum))
        ranking.append(docs[i])
        remain[i] = -np.inf
        np.maximum(maximum, sim[i], out=maximum)
    return ranking


def score(args):
    manifest = load(args.manifest)['queries']
    report = load(args.budgeted)
    cached = {'oof': {}, 'dev': index(load(args.dev8b))}
    for p in args.oof8b:
        cached['oof'].update(index(load(p)))
    base = {'oof': index(load(args.oof_frozen)), 'dev': index(load(args.dev_frozen))}
    evaluations = []
    for domain in sorted({r['domain'] for r in manifest}):
        records = [r for r in manifest if r['domain'] == domain]
        required = {d for r in records for d in r['docs']}
        content = {}
        with (args.data_root / domain / 'documents.jsonl').open(encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line)
                if doc['id'] in required:
                    content[doc['id']] = doc['content']
                    if content.keys() == required:
                        break
        if content.keys() != required:
            raise ValueError(f'Incomplete corpus for {domain}')
        gold = {s: read_qrels(args.data_root / domain / f'qrels_{s}.txt') for s in ('train', 'dev')}
        for row in records:
            q, docs = row['qid'], row['docs']
            split = 'oof' if row['split'] == 'train' else 'dev'
            key = domain, q
            weights = dict(base[split][key]['frozen_scores'])
            model = cached[split][key]['scores']
            values = .7 * z([weights[d] for d in docs]) + .3 * z([model[d] for d in docs])
            terms_query = terms(row['query'])
            sets = [shingles(passages(content[d], terms_query)[0]) for d in docs]
            sim = similarities(sets)
            arms = {str(w): greedy(docs, values, sim, w) for w in PENALTIES}
            scores = {k: ndcg10(v, gold[row['split']][q]) for k, v in arms.items()}
            edges = [[docs[i], docs[j], float(sim[i, j])]
                     for i, j in zip(*np.nonzero(np.triu(sim, k=1)))]
            evaluations.append({'domain': domain, 'qid': q, 'split': split, 'cohort': row['cohort'],
                                'pairs': len(edges), 'near_duplicate_edges': edges,
                                'rankings': arms, 'scores': scores})
        print(domain, len(evaluations), 'queries scored', flush=True)
    return evaluations, report


def analyze(rows, archive, reference, selected):
    indexed = {(r['domain'], r['qid']): r for r in rows}
    score = {}
    for name in (*map(str, PENALTIES), 'selected'):
        deltas = []
        for r in archive:
            d, q = r['domain'], r['qid']
            row = indexed.get((d, q))
            delta = 0.
            if row is not None:
                base = row['scores']['0.0']
                if abs(base-r['scores']['dense_24']) > 1e-12:
                    raise ValueError(f'Reference differs for {d}/{q}')
                key = selected[row['cohort']] if name == 'selected' else name
                delta = row['scores'][key] - base
            deltas.append((d, delta))
        paired = paired_delta(deltas)
        score[name] = {'ndcg': reference + paired['point'], 'change': paired}
    return score


def choose(rows):
    return max(map(str, PENALTIES), key=lambda a: macro(
        (r['domain'], r['scores'][a]-r['scores']['0.0']) for r in rows
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-root', 'budgeted', 'protected', 'dev8b', 'oof-frozen', 'dev-frozen', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--oof8b', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    rows, archive = score(args)
    protected = load(args.protected)
    train = [r for r in rows if r['split'] == 'oof']
    test = [r for r in rows if r['split'] == 'dev']
    chosen = choose(train)
    cross = {i: choose([r for r in train if r['cohort'] != i]) for i in (0, 1)}
    result = {'chosen': chosen, 'crossfit': cross, 'per_query': rows}
    for split, group, selected in (('oof', train, cross), ('dev', test, {0: chosen})):
        result[split] = analyze(group, archive[f'{split}_per_query'],
                                protected[split]['full_policy']['1']['ndcg'], selected)
        print(split, sum(r['pairs'] > 0 for r in group), 'near-duplicate queries', result[split], flush=True)
    print('Selected', chosen, cross)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, separators=(',', ':'))
        f.write('\n')


if __name__ == '__main__':
    main()
