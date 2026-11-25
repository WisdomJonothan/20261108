"""Separate score shape from order and test a fixed two-reranker ensemble."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges, choose, grade, summarize
from replay_anchor_cap import z
from replay_anchor_fallback import index


RANKERS = ('qwen8b', 'yuanbao')
ENSEMBLES = tuple(f'mean_{width}_{weight}' for width in (30, 60) for weight in (35, 50, 65))
POLICIES = ('current', *ENSEMBLES)


def swap_shape(values, donor):
    order = np.argsort(-values, kind='stable')
    levels = np.sort(donor)[::-1]
    # Scores originate from BF16. Break donor ties while preserving the first scorer's order.
    levels = levels - 1e-10 * np.arange(len(levels))
    swapped = np.empty_like(values)
    swapped[order] = levels
    return z(swapped)


def rank(docs, scores):
    return [docs[index] for index in np.argsort(-scores, kind='stable')[:10]]


def replay(docs, left, right, current):
    vectors = {name: np.array([scores[doc] for doc in docs], dtype=float)
               for name, scores in zip(RANKERS, (left, right), strict=True)}
    ranks = {'current': current}
    prior30 = z(1 / np.log2(2 + np.arange(30)))
    for name, values in vectors.items():
        ranks[f'{name}/standalone'] = rank(docs[:30], values[:30])
        for donor, shape in vectors.items():
            scores = z(values[:30]) if donor == name else swap_shape(values[:30], shape[:30])
            ranks[f'{name}/shape_{donor}'] = rank(docs[:30], .5 * prior30 + .5 * scores)
    for width in (30, 60):
        combined = np.stack([z(values[:width]) for values in vectors.values()]).mean(axis=0)
        prior = z(1 / np.log2(2 + np.arange(width)))
        for weight in (35, 50, 65):
            fraction = weight / 100
            ranks[f'mean_{width}_{weight}'] = rank(docs[:width], (1 - fraction) * prior + fraction * combined)
    return ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'reference', 'data-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--qwen8b', type=Path, nargs='+', required=True)
    parser.add_argument('--yuanbao', type=Path, nargs='+', required=True)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')

    split = 'dev' if args.selection else 'train'
    queries = index({'per_query': load(args.manifest)['queries']})
    references = index(load(args.reference))
    pools = [candidate_judges(paths)[0] for paths in (args.qwen8b, args.yuanbao)]
    if any(len(pool) != 1 for pool in pools):
        raise ValueError('Each scorer must refer to a single checkpoint and prompt')
    score_sets = [next(iter(pool.values())) for pool in pools]
    if len(queries) != (519 if split == 'dev' else 186) or any(s.keys() != queries.keys() for s in score_sets):
        raise ValueError('Unexpected set of matched queries')
    if references.keys() != queries.keys():
        raise ValueError('Ranking reference uses a different query set')
    rows = []
    for key, row in queries.items():
        docs = row['docs']
        left, right = [s[key]['scores'] for s in score_sets]
        if len(set(docs)) != 60 or left.keys() != set(docs) or right.keys() != set(docs):
            raise ValueError(f'Models did not score the same 60 documents for {key}')
        rankings = replay(docs, left, right, references[key]['rankings']['current'])
        rows.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                    {'rankings': rankings})
    rows = grade(rows, split, args.data_root)

    if args.selection:
        selected = load(args.selection)['selected']
        if selected not in POLICIES:
            raise ValueError('Official selection must use a permitted training policy')
        summary = summarize(rows, selected)
        report = {'dev': summary}
    else:
        selected = choose(rows, POLICIES)
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], POLICIES)
                   for f in sorted({r['cohort'] for r in rows})}
        summary = summarize(rows, selected, by_fold)
        report = {'training': summary, 'by_fold': by_fold,
                  'original_65': summarize([r for r in rows if r['actual_reference']], selected, by_fold)}

    report.update({'selected': selected, 'per_query': rows})
    for comparator in ('current', 'selected'):
        gains = summary[comparator]['change']['per_domain']
        smallest = min(mean(v for d, v in gains.items() if d != held) for held in gains)
        print(comparator, selected, summary[comparator]['score'], summary[comparator]['change']['point'], smallest)
    print({arm: round(value['score'], 7) for arm, value in summary.items()})
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
