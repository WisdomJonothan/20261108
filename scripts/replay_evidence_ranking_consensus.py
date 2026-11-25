"""Test whether agreeing rankers can make better edits around a frozen ranking."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import choose, grade, summarize
from eval_rerank_headroom_study import diagnose
from replay_anchor_fallback import index


ARMS = ('current', 'verified', 'reasoned', 'pointwise',
        'agreement_verified', 'agreement_reasoned', 'agreement_all', 'conditional_head')
EXTENSIONS = ('agreement_offset2', 'agreement_offset15', 'agreement_offset60',
              'agreement_add_wide', 'agreement_add_listwise', 'agreement_add_both')
PAIR_POLICIES = ('mean_position', 'pair_majority', 'pair_vote_margin')


def reciprocal(lists, offset=5):
    scores = {}
    for ranking in lists:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0) + 1 / (rank + offset)
    return sorted(scores, key=lambda doc: -scores[doc])[:10]


def pair_orders(lists):
    candidates = list(dict.fromkeys(d for ranking in lists for d in ranking))
    positions = [{d: i for i, d in enumerate(ranking)} for ranking in lists]
    means = {d: sum(book.get(d, len(book)) for book in positions) for d in candidates}
    wins, margins = {}, {}
    for d in candidates:
        net = [sum((book.get(other, len(book)) > book.get(d, len(book))) -
                   (book.get(other, len(book)) < book.get(d, len(book))) for book in positions)
               for other in candidates]
        wins[d] = sum((margin > 0) - (margin < 0) for margin in net)
        margins[d] = sum(margin * abs(margin) for margin in net)
    return {'mean_position': sorted(candidates, key=means.__getitem__)[:10],
            'pair_majority': sorted(candidates, key=lambda d: (-wins[d], means[d]))[:10],
            'pair_vote_margin': sorted(candidates, key=lambda d: (-margins[d], means[d]))[:10]}


def replay(reasoner, pointwise, listwise=None, compare_pairs=False):
    reasoner, pointwise = index(load(reasoner)), index(load(pointwise))
    if reasoner.keys() != pointwise.keys():
        raise ValueError('Pointwise and evidence rankings require the same queries')
    if listwise:
        listwise = index(load(listwise))
        if listwise.keys() != reasoner.keys():
            raise ValueError('Listwise rankings require the same queries')
    rows = []
    for key, row in reasoner.items():
        r, p = row['rankings'], pointwise[key]['rankings']
        current, reasoned, verified = (r[a] for a in ('current', 'reasoned/0.5', 'verified/0.5'))
        if current != p['current']:
            raise ValueError(f'Original rankings disagree for {key}')
        point = p['pretrained/adaptive30_free']
        votes = current, reasoned, verified, point
        combined = reciprocal(votes)
        anchor = reasoned[0] if reasoned[0] == point[0] else current[0]
        ranks = {'current': current, 'reasoned': reasoned, 'verified': verified, 'pointwise': point,
                 'agreement_verified': reciprocal((current, verified, point)),
                 'agreement_reasoned': reciprocal((current, reasoned, point)),
                 'agreement_all': combined,
                 'conditional_head': [anchor, *[d for d in combined if d != anchor][:9]]}
        if listwise:
            record = listwise[key]['rankings']
            if record['current'] != current:
                raise ValueError(f'Listwise reference rankings disagree for {key}')
            wide = p['pretrained/adaptive60_protected']
            joint = record['pretrained/start_from_repair_m25']
            ranks.update({f'agreement_offset{offset}': reciprocal(votes, offset) for offset in (2, 15, 60)})
            ranks.update({'agreement_add_wide': reciprocal((*votes, wide)),
                          'agreement_add_listwise': reciprocal((*votes, joint)),
                          'agreement_add_both': reciprocal((*votes, wide, joint))})
        if compare_pairs:
            ranks.update(pair_orders(votes))
        rows.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                    {'rankings': ranks})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reasoner', 'pointwise', 'data-root', 'output', 'selection', 'listwise'):
        parser.add_argument(f'--{name}', type=Path, required=name not in ('selection', 'listwise'))
    parser.add_argument('--compare-pairs', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    split = 'dev' if args.selection else 'train'
    rows = grade(replay(args.reasoner, args.pointwise, args.listwise, args.compare_pairs), split, args.data_root)
    policies = (*ARMS, *EXTENSIONS) if args.listwise else ARMS
    if args.compare_pairs:
        policies += PAIR_POLICIES
    if len(rows) != (519 if split == 'dev' else 186):
        raise ValueError('Unexpected evaluation population')
    if args.selection:
        selected = load(args.selection)['selected']
        if selected not in policies:
            raise ValueError('The selected training policy needs its original inputs')
        summary = summarize(rows, selected)
        if abs(summary['current']['score'] - .3958942538410169) > 1e-12:
            raise ValueError('Unexpected official dev reference')
        report = {'dev': summary, 'diagnostics': diagnose(rows, selected, args.data_root, 'current')}
    else:
        selected = choose(rows, policies)
        folds = sorted({r['cohort'] for r in rows})
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], policies) for f in folds}
        summary = summarize(rows, selected, by_fold)
        report = {'training': summary, 'by_fold': by_fold,
                  'original_65': summarize([r for r in rows if r['actual_reference']], selected, by_fold)}
    report.update({'selected': selected, 'per_query': rows})
    print(selected, {arm: round(v['score'], 8) for arm, v in summary.items()})
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
