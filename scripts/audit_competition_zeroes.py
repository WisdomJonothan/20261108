"""Locate zero-score queries after the frozen six-pair competition policy."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from coverage_cf_bound import DOMAINS, read_qrels
from lib_reteco import read_trec_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-actions', action='store_true')
    args = parser.parse_args()
    data = args.data_root or args.root / 'data/track1_tempo'
    report = json.loads((args.root / 'results/analysis/position_oracle_dev_20261124.json')
                        .read_text(encoding='utf-8'))
    historical = {}
    if args.trace_actions:
        art = args.root / 'results/replacement_v2'
        cache = json.loads((art / 'expanded3_fixed_dev.json').read_text(encoding='utf-8'))
        historical = {(r['domain'], r['qid']): r['actions']
                      for r in cache['budgets']['18']['per_query']}
    rows = []
    totals = Counter()
    per_domain = {}
    for domain in DOMAINS:
        dense = read_trec_run(args.root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        gold = read_qrels(data / domain / 'qrels_dev.txt')
        if historical:
            ce = read_trec_run(args.root / 'runs' /
                               f'fusion100_full_ce_s17_20261115_{domain}_dev_ce.trec')
            frozen = read_trec_run(args.root / 'runs' / f'sub_sft_e2bc_{domain}_dev_head10.trec')
        counts = Counter()
        for row in (r for r in report['per_query'] if r['domain'] == domain):
            qid = row['qid']
            base = row['baseline']
            policy = base + row['policies']['external6']['delta']
            counts['queries'] += 1
            counts['recovered'] += base <= 1e-12 and policy > 1e-12
            counts['new_zero'] += base > 1e-12 and policy <= 1e-12
            if policy > 1e-12:
                continue
            counts['zero'] += 1
            rank = min((rank for rank, _, doc in dense[qid] if doc in gold[qid]), default=math.inf)
            bucket = 'top100' if rank <= 100 else '101_to_1000' if rank <= 1000 else 'missing_top1000'
            counts[bucket] += 1
            item = {'domain': domain, 'qid': qid, 'dense_first_positive_rank':
                    int(rank) if math.isfinite(rank) else None, 'bucket': bucket}
            if historical and bucket == 'top100':
                top = {doc for _, _, doc in frozen[qid]}
                outside = sorted((r, view, doc) for view, run in enumerate((dense[qid], ce[qid]))
                                 for r, _, doc in run[:100] if doc not in top)
                pool20 = list(dict.fromkeys(doc for _, _, doc in outside))[:20]
                actions18 = historical[domain, qid]
                actions6 = row['actions']['external']
                assert set(a['challenger'] for a in actions18) <= set(pool20)
                stage_flags = {
                    'candidate20': bool(set(pool20) & gold[qid]),
                    'prior18_positive_challenger': any(a['challenger'] in gold[qid] for a in actions18),
                    'judged6_beneficial': any(a['delta'] > 1e-12 for a in actions6),
                    'accepted_beneficial': any(a['delta'] > 1e-12 and a['agree'] and a['mp'] > .5
                                               for a in actions6),
                }
                assert list(stage_flags.values()) == sorted(stage_flags.values(), reverse=True)
                item['stages'] = stage_flags
                for stage, present in stage_flags.items():
                    counts[stage] += present
            rows.append(item)
        per_domain[domain] = dict(counts)
        totals.update(counts)
    assert totals['queries'] == 519
    assert totals['zero'] == 138 - totals['recovered'] + totals['new_zero']
    result = {'starting_system': report['summary']['arms']['external6']['ndcg'],
              'scope': 'Official dev. Judge outcome replayed from the fixed competition cache. '
              'Zero-score queries are classified by the judged-positive ranks in the dense run.',
              'counts': totals, 'per_domain': per_domain, 'zero_queries': rows}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
