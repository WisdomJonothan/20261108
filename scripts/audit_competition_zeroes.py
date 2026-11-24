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
    args = parser.parse_args()
    data = args.data_root or args.root / 'data/track1_tempo'
    report = json.loads((args.root / 'results/analysis/position_oracle_dev_20261124.json')
                        .read_text(encoding='utf-8'))
    rows = []
    totals = Counter()
    per_domain = {}
    for domain in DOMAINS:
        dense = read_trec_run(args.root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        gold = read_qrels(data / domain / 'qrels_dev.txt')
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
            rows.append({'domain': domain, 'qid': qid, 'dense_first_positive_rank':
                         int(rank) if math.isfinite(rank) else None, 'bucket': bucket})
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
