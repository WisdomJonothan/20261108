"""Attribute training-side nDCG headroom to successive candidate boundaries."""
import argparse
from functools import cache
import json
from pathlib import Path

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_fallback import index


def ideal(pool, gold):
    return ndcg10(sorted(set(pool) & gold), gold)


def audit(args):
    pools = load(args.pools)
    shortlisted = index({'per_query': load(args.shortlist)['queries']})
    if set(shortlisted) != set(index(pools)):
        raise ValueError('The shortlist must cover the same query IDs as the wider pools')
    if {row['split'] for row in pools['per_query']} != {'train'}:
        raise ValueError('The headroom audit uses official training judgments only')

    @cache
    def sources(domain):
        return tuple(read_trec_run(root / f'dense_{name}_{domain}_train.trec')
                     for name, root in (('mxbai', args.mxbai), ('bge', args.bge)))

    depth = pools['models']['source_depth']
    gold_by_domain = {domain: read_qrels(args.data_root / domain / 'qrels_train.txt')
                      for domain in {row['domain'] for row in pools['per_query']}}
    output = []
    for row in pools['per_query']:
        domain, qid = row['domain'], row['qid']
        gold = gold_by_domain[domain][qid]
        original = set(row['original'])
        top_ten = row['prior_ten']
        shortlist = set(shortlisted[domain, qid]['docs'])
        union = original | {doc for run in sources(domain) for _, _, doc in run[qid][:depth]}
        if not set(top_ten) <= shortlist <= original <= union or len(union) != row['union_count']:
            raise ValueError(f'Unexpected candidate nesting or source depth for {domain}/{qid}')
        levels = {'actual': ndcg10(top_ten, gold), 'top_ten': ideal(top_ten, gold),
                  'shortlist': ideal(shortlist, gold), 'original': ideal(original, gold),
                  'broader_union': ideal(union, gold), 'all_qrels': ideal(gold, gold)}
        output.append({'domain': domain, 'qid': qid, 'scores': levels})

    levels = tuple(output[0]['scores'])
    domains = sorted(gold_by_domain)
    summaries = {}
    for group in ('macro', *domains):
        rows = [r for r in output if group == 'macro' or r['domain'] == group]
        ceilings = {level: macro((r['domain'], r['scores'][level]) for r in rows) for level in levels}
        summaries[group] = {
            'queries': len(rows), 'ceilings': ceilings,
            'increments': {right: ceilings[right] - ceilings[left]
                           for left, right in zip(levels[:-1], levels[1:], strict=True)},
            'zero_with_evidence_by_stage': {level: sum(r['scores']['actual'] == 0 and
                                                       r['scores'][level] > 0 for r in rows)
                                            for level in levels[1:]},
        }
    return {'source_depth': depth, 'summary': summaries, 'per_query': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pools', 'shortlist', 'data-root', 'mxbai', 'bge', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    result = audit(args)
    with args.output.open('x', encoding='utf8') as file:
        json.dump(result, file, separators=(',', ':'))
    print(json.dumps(result['summary']['macro'], indent=2))


if __name__ == '__main__':
    main()
