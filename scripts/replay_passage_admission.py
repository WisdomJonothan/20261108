"""Test one evidence-confirmed admission from a new candidate source."""
import argparse
import json
from math import log2
from pathlib import Path
from statistics import mean

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges, choose, grade, summarize
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


SOURCES = ('dense_only', 'facet_span')
ARMS = ('current', *(f'{source}/{name}' for source in SOURCES for name in ('tail', 'nonhead')))


def prepare(args):
    rows = index({'per_query': load(args.manifest)['queries']})
    pools = index(load(args.pools))
    if rows.keys() != pools.keys():
        raise ValueError('The candidate sources must share the same queries')
    candidates = []
    for key, r in rows.items():
        orders = pools[key]['pools']
        if orders['current'] != r['docs'] or len(set(r['docs'])) != 60:
            raise ValueError(f'The frozen candidate pool differs for {key}')
        previous = set(r['docs'])
        proposals = {source: [d for d in orders[source] if d not in previous][:4] for source in SOURCES}
        docs = list(dict.fromkeys(d for source in SOURCES for d in proposals[source]))
        if any(len(items) != 4 for items in proposals.values()):
            raise ValueError(f'Each matched retrieval arm needs four candidates for {key}')
        candidates.append(r | {'docs': docs, 'proposals': proposals})
    return {'queries': candidates}


def replacement(prior, incumbent, contender, old_pointwise, new_pointwise, allowed, first):
    output = prior.copy()
    strongest = 0.
    for i in range(first, 10):
        left = prior[i]
        if left not in incumbent:
            continue
        old = incumbent[left]
        old_grade = old['reasoned'] if old['verified'] else old['direct']
        for right in allowed:
            new = contender[right]
            expected = new['reasoned'] - old_grade
            if (not new['verified'] or new['reasoned'] < 2. or expected < .5
                    or new_pointwise[right] <= old_pointwise[left]):
                continue
            impact = expected / log2(i+2)
            if impact > strongest:
                strongest = impact
                output = prior.copy()
                output[i] = right
    return output


def observations(paths):
    copies = [r for f in paths for r in load(f)['per_query']]
    return {k: row['observations'] for k, row in index({'per_query': copies}).items()}


def replay(args):
    queries = index({'per_query': load(args.manifest)['queries']})
    old_evidence, new_evidence = (observations(p) for p in (args.old_evidence, args.new_evidence))
    old_judge, new_judge = (candidate_judges(p)[0]['pretrained']
                           for p in (args.old_pointwise, args.new_pointwise))
    head = index(load(args.reference))
    if not all(q.keys() == queries.keys() for q in (old_evidence, new_evidence, old_judge, new_judge, head)):
        raise ValueError('Every prediction source must include the same query IDs')
    records = []
    for key, r in queries.items():
        start = head[key]['rankings']['verified_0.5']
        left, right = old_evidence[key], new_evidence[key]
        point_left, point_right = (judge[key]['scores'] for judge in (old_judge, new_judge))
        if right.keys() != set(r['docs']) or point_right.keys() != right.keys() or not set(start) <= point_left.keys():
            raise ValueError(f'The evidence assessments differ from the proposed candidate IDs for {key}')
        orders = {'current': start}
        for source in SOURCES:
            proposals = r['proposals'][source]
            for name, first in (('tail',5), ('nonhead',2)):
                orders[f'{source}/{name}'] = replacement(start, left, right, point_left, point_right, proposals, first)
        records.append({k: r[k] for k in ('domain','qid','cohort','split','actual_reference')} |
                       {'rankings': orders})
    return records


def evaluate(args):
    split = 'dev' if args.selection and not args.confirmation else 'train'
    rows = grade(replay(args), split, args.data_root)
    if len(rows) != (155 if args.confirmation else (519 if split == 'dev' else 186)):
        raise ValueError('Unexpected number of evaluation queries')
    if args.selection:
        selected = load(args.selection)['selected']
        summary = summarize(rows, selected)
        if split == 'dev' and abs(summary['current']['score']-.40674178096988606) > 1e-12:
            raise ValueError('Unexpected official-dev reference')
        report = {'confirmation' if args.confirmation else 'dev': summary}
    else:
        selected = choose(rows, ARMS)
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], ARMS)
                   for f in sorted({r['cohort'] for r in rows})}
        summary = summarize(rows, selected, by_fold)
        report = {'training': summary, 'by_fold': by_fold}
    domain = summary['selected']['change']['per_domain']
    report.update({'selected':selected, 'per_query':rows,
                   'selected_minimum_leave_one_domain_out':min(
                       mean(v for d,v in domain.items() if d!=omit) for omit in domain)})
    report['matched_source_effect'] = {
        name: paired_delta([(r['domain'],r['scores'][f'facet_span/{name}']-r['scores'][f'dense_only/{name}'])
                            for r in rows]) for name in ('tail','nonhead')}
    print(selected,{k:round(v['score'],8) for k,v in summary.items()},flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    steps=parser.add_subparsers(dest='step',required=True)
    setup, assessment = (steps.add_parser(name) for name in ('prepare','evaluate'))
    for step in (setup,assessment):
        for key in ('manifest','output'):
            step.add_argument(f'--{key}',type=Path,required=True)
    setup.add_argument('--pools',type=Path,required=True)
    for name in ('reference','data-root','selection'):
        assessment.add_argument(f'--{name}',type=Path,required=name!='selection')
    assessment.add_argument('--confirmation',action='store_true')
    for name in ('old-evidence','new-evidence','old-pointwise','new-pointwise'):
        assessment.add_argument(f'--{name}',type=Path,nargs='+',required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.step=='evaluate' and args.confirmation and not args.selection:
        parser.error('Confirmation requires a previously selected rule')
    result=(prepare if args.step=='prepare' else evaluate)(args)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(result,file,separators=(',',':'))


if __name__=='__main__':
    main()
