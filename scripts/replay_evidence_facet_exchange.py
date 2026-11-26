"""Test whether candidate evidence complements the rest of the top ten."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import choose, grade, summarize
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


ARMS = ('current', 'pointwise', 'coverage', 'coverage_required')


def exchanges(start, outsiders, category, n_facets):
    if n_facets < 2:
        return dict.fromkeys(ARMS, start)
    names = [*start, *outsiders]
    active = [*range(n_facets + 1), 5]
    logits = np.array([category[doc] for doc in names])[:, active]
    probability = np.exp(logits - np.logaddexp.reduce(logits, axis=1)[:, None])
    support = 1. - probability[:, 0]
    by_facet = probability[:, 1:-1]
    coverage = by_facet[:10].max(axis=0)
    choices = dict.fromkeys(ARMS, start)
    best = dict.fromkeys(ARMS, 0.)
    for rank in range(5, 10):
        other = np.delete(by_facet[:10], rank, axis=0).max(axis=0)
        for j, doc in enumerate(outsiders, 10):
            advantage = support[j] - support[rank]
            if support[j] < .55 or advantage < .02:
                continue
            gain = advantage / np.log2(rank + 2)
            novelty = (np.maximum(other, by_facet[j]) - coverage).mean()
            effects = {'pointwise': gain, 'coverage': gain + .5*novelty,
                       'coverage_required': gain + .5*novelty if novelty > 0 else 0.}
            for key, value in effects.items():
                if value > best[key]:
                    best[key] = value
                    choices[key] = [*start[:rank], doc, *start[rank+1:]]
    return choices


def prepare(args):
    reference = index(load(args.reference))
    source = index({'per_query': load(args.manifest)['queries']})
    logits = index({'per_query': [r for f in args.categories for r in load(f)['per_query']]})
    if not reference.keys() == source.keys() == logits.keys():
        raise ValueError('The fixed rankings, candidate manifest and proof outputs must align')
    output = []
    for key, r in source.items():
        start = reference[key]['rankings']['verified_0.5']
        outsiders = [d for d in r['docs'] if d not in start]
        row = logits[key]
        if len(start) != len(set(start)) or row['categories'].keys() != set(start + outsiders):
            raise ValueError(f'Need scores for all incumbent and candidate documents for {key}')
        ranks = exchanges(start, outsiders, row['categories'], row['facet_count'])
        output.append({k: r[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                      {'rankings': ranks, 'facet_count': row['facet_count']})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'manifest', 'data-root', 'output', 'selection'):
        parser.add_argument(f'--{name}', type=Path, required=name != 'selection')
    parser.add_argument('--categories', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    split = 'dev' if args.selection else 'train'
    rows = grade(prepare(args), split, args.data_root)
    if len(rows) != (519 if split == 'dev' else 186):
        raise ValueError('Unexpected split size')
    if args.selection:
        selected = load(args.selection)['selected']
        if selected not in ARMS:
            raise ValueError('Unexpected selected arm')
        summary = summarize(rows, selected)
        if abs(summary['current']['score'] - .40674178096988606) > 1e-12:
            raise ValueError('Expected the fixed official dev reference')
        report = {'dev': summary}
    else:
        selected = choose(rows, ARMS)
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], ARMS)
                   for f in sorted({r['cohort'] for r in rows})}
        summary = summarize(rows, selected, by_fold)
        report = {'training': summary, 'by_fold': by_fold,
                  'original_65': summarize([r for r in rows if r['actual_reference']], selected, by_fold)}
    report.update({'selected': selected, 'per_query': rows,
                   'eligible_queries': sum(r['facet_count'] >= 2 for r in rows)})
    for method in ('coverage', 'coverage_required'):
        report[f'{method}_versus_pointwise'] = paired_delta([
            (r['domain'], r['scores'][method]-r['scores']['pointwise']) for r in rows])
    deltas = summary['selected']['change']['per_domain']
    report['selected_minimum_leave_one_domain_out'] = min(
        mean(v for d, v in deltas.items() if d != held) for held in deltas)
    print(selected, {name: round(values['score'], 8) for name, values in summary.items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
