"""Allow a head swap when both reasoning grades and pointwise ranks support it."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import choose, grade, summarize
from eval_rerank_headroom_study import diagnose
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


ARMS = ('candidate', *(f'{mode}_{margin:g}' for mode in ('reasoned', 'verified')
                       for margin in (.5, 1., 1.5)))


def head(ranking, judges, estimates, use_verified, margin):
    eligible = [d for d in ranking[1:5] if d in judges[:5] and d in estimates
                and (not use_verified or estimates[d]['verified'])]
    first = ranking[0]
    if not eligible or first not in estimates:
        return ranking
    best = max(eligible, key=lambda d: estimates[d]['reasoned'])
    best_grade = estimates[best]['reasoned']
    if best_grade < 2 or best_grade - estimates[first]['reasoned'] < margin:
        return ranking
    revised = ranking.copy()
    position = revised.index(best)
    revised[0], revised[position] = revised[position], revised[0]
    return revised


def replay(reference, archive):
    ranks = load(reference)
    measured = index({'per_query': [r for file in archive for r in load(file)['per_query']]})
    if measured.keys() != index(ranks).keys():
        raise ValueError('Consensus and evidence scores must cover the same questions')
    rows = []
    for row in ranks['per_query']:
        source = row['rankings']
        candidate = source['agreement_offset60']
        obs = measured[row['domain'], row['qid']]['observations']
        rankings = {'current': source['current'], 'candidate': candidate}
        for method in ARMS[1:]:
            mode, margin = method.split('_')
            rankings[method] = head(candidate, source['pointwise'], obs, mode == 'verified', float(margin))
        rows.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                    {'rankings': rankings})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'data-root', 'output', 'selection'):
        parser.add_argument(f'--{name}', type=Path, required=name != 'selection')
    parser.add_argument('--evidence', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    split = 'dev' if args.selection else 'train'
    rows = grade(replay(args.reference, args.evidence), split, args.data_root)
    if len(rows) != (519 if split == 'dev' else 186):
        raise ValueError('Unexpected evaluation population')
    if args.selection:
        selected = load(args.selection)['selected']
        if selected not in ARMS:
            raise ValueError('Unexpected selection')
        summary = summarize(rows, selected)
        report = {'dev': summary, 'diagnostics': diagnose(rows, selected, args.data_root, 'current')}
    else:
        selected = choose(rows, ARMS)
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], ARMS)
                   for f in sorted({r['cohort'] for r in rows})}
        summary = summarize(rows, selected, by_fold)
        report = {'training': summary, 'by_fold': by_fold}
    comparison = paired_delta([(r['domain'], r['scores'][selected] - r['scores']['candidate']) for r in rows])
    comparison['changed_rank_one'] = sum(r['rankings'][selected][0] != r['rankings']['candidate'][0]
                                         for r in rows)
    report['fixed_choice_against_consensus'] = comparison
    report.update({'selected': selected, 'per_query': rows})
    print(selected, {k: round(v['score'], 8) for k, v in summary.items()})
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
