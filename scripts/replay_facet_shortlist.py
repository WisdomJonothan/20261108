"""Compare complementary evidence selection with the same fourteen grading slots."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import choose, grade, summarize
from eval_temporal_evidence_categories import ranking
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_facet_exchange import distribution
from replay_evidence_ranking_consensus import reciprocal


ARMS = ('current', 'pointwise_selector', 'coverage_selector')


def choose_pool(docs, logits, k, use_coverage):
    if k < 2:
        return docs[:14]
    support, facets = distribution(docs, logits, k)
    covered = facets[:10].max(axis=0)
    near = [10, 11, *range(14, 18)]
    deep = [12, 13, *range(18, 22)]
    selected = list(range(10))
    for tier in (near, near, deep, deep):
        def utility(i):
            improvement = np.maximum(facets[i]-covered, 0.).mean()
            return support[i] + (.5*improvement if use_coverage else 0.)
        choice = max(tier, key=utility)
        selected.append(choice)
        tier.remove(choice)
        covered = np.maximum(covered, facets[choice])
    return [docs[i] for i in selected]


def evidence_rank(docs, grades, current, pointwise):
    reasoned = ranking(docs, [grades[d]['reasoned'] for d in docs])
    checked = ranking(docs, [grades[d]['reasoned'] if grades[d]['verified'] else grades[d]['direct']
                             for d in docs])
    consensus = reciprocal((current, reasoned, checked, pointwise), 60)
    return head(consensus, pointwise, {d: grades[d] for d in docs}, True, .5)


def extra_grading_pool(docs, logits, facet_count):
    selected = set(choose_pool(docs, logits, facet_count, False))
    selected.update(choose_pool(docs, logits, facet_count, True))
    return [d for d in docs[14:] if d in selected]


def gather(paths):
    rows = {}
    for path in paths:
        for row in load(path)['per_query']:
            key = row['domain'], row['qid']
            value = rows.setdefault(key, {})
            scores = row['observations']
            if not scores.keys().isdisjoint(value):
                raise ValueError(f'Reasoner scores are repeated for {key}')
            value.update(scores)
    return rows


def replay(args):
    questions = load(args.manifest)['queries']
    ranking_rows = index(load(args.consensus))
    originals = index(load(args.reference))
    logits = index({'per_query': [r for f in args.categories for r in load(f)['per_query']]})
    grades = gather(args.grades)
    keys = {(q['domain'], q['qid']) for q in questions}
    if not all(item.keys() == keys for item in (ranking_rows, originals, logits, grades)):
        raise ValueError('The candidate pools and scores must share the exact same questions')
    result = []
    for q in questions:
        key = q['domain'], q['qid']
        docs = q['docs'][:22]
        original, pointwise = (ranking_rows[key]['rankings'][a] for a in ('current', 'pointwise'))
        current = originals[key]['rankings']['verified_0.5']
        if (len(set(docs)) != 22 or not set(docs[:14]) <= grades[key].keys() <= set(docs)
                or docs[:10] != original):
            raise ValueError(f'The stored evidence inputs differ for {key}')
        if evidence_rank(docs[:14], grades[key], original, pointwise) != current:
            raise ValueError(f'The frozen rankings differ for {key}')
        ranked = {'current': current}
        row = logits[key]
        for use_coverage, arm in ((False, 'pointwise_selector'), (True, 'coverage_selector')):
            candidates = choose_pool(docs, row['categories'], row['facet_count'], use_coverage)
            if not set(candidates) <= grades[key].keys():
                raise ValueError(f'Missing selected evidence grades for {key}')
            ranked[arm] = evidence_rank(candidates, grades[key], original, pointwise)
        result.append({k: q[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                      {'rankings': ranked})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'reference', 'consensus', 'data-root', 'output', 'selection'):
        parser.add_argument(f'--{name}', type=Path, required=name != 'selection')
    parser.add_argument('--confirmation', action='store_true')
    for name in ('grades', 'categories'):
        parser.add_argument(f'--{name}', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.confirmation and args.selection is None:
        parser.error('Confirmation needs the fixed selection')
    split = 'dev' if args.selection and not args.confirmation else 'train'
    rows = grade(replay(args), split, args.data_root)
    expected = 155 if args.confirmation else (519 if split == 'dev' else 186)
    if len(rows) != expected:
        raise ValueError('Unexpected query population')
    if args.selection:
        selected = load(args.selection)['selected']
        if selected not in ARMS:
            raise ValueError('Unexpected fixed training choice')
        summary = summarize(rows, selected)
        if split == 'dev' and abs(summary['current']['score'] - .40674178096988606) > 1e-12:
            raise ValueError('Expected the frozen official dev reference')
        report = {'confirmation' if args.confirmation else 'dev': summary}
    else:
        selected = choose(rows, ARMS)
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], ARMS)
                   for f in sorted({r['cohort'] for r in rows})}
        summary = summarize(rows, selected, by_fold)
        report = {'by_fold': by_fold, 'training': summary}
    deltas = summary['selected']['change']['per_domain']
    report.update({'selected': selected, 'per_query': rows,
                   'selected_minimum_leave_one_domain_out': min(
                       mean(v for d, v in deltas.items() if d != held) for held in deltas),
                   'coverage_versus_pointwise': paired_delta([
                       (r['domain'], r['scores']['coverage_selector']-r['scores']['pointwise_selector']) for r in rows])})
    print(selected, {a: round(v['score'], 8) for a, v in summary.items()}, flush=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
