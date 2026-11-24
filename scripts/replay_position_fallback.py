"""Evaluate a CPU position exchange on queries where the six-pair judge abstains."""
import argparse
import json
from pathlib import Path

from benchmark_pairwise_replacement_v2 import domain_macro, paired_bootstrap


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=Path('results/analysis'))
    args = parser.parse_args()
    ranker = load(args.analysis / 'position_rank_features_20261124.json')
    reports = {}
    for split in ('oof', 'dev'):
        external = load(args.analysis / f'position_oracle_{split}_20261124.json')['per_query']
        rank = {(r['domain'], r['qid']): r for r in ranker['per_query'][split]}
        assert {(r['domain'], r['qid']) for r in external} == set(rank)
        rows = []
        for query in external:
            domain, qid = query['domain'], query['qid']
            base = query['baseline']
            assert abs(base - rank[domain, qid]['baseline']) < 1e-12
            decision = query['policies']['external6']
            fallback = decision['chosen'] is None
            extra = rank[domain, qid]['one_exchange'] - base if fallback else 0.
            rows.append({'domain': domain, 'qid': qid, 'external6': base + decision['delta'],
                         'fallback_score': base + decision['delta'] + extra,
                         'eligible': fallback, 'delta': extra})
        reports[split] = {'queries': len(rows), 'fallback_eligible': sum(r['eligible'] for r in rows),
                          'external6': domain_macro(rows, key='external6'),
                          'fallback_score': domain_macro(rows, key='fallback_score'),
                          'fallback_extra_delta': domain_macro(rows, key='delta'),
                          'bootstrap': paired_bootstrap(rows)}
    path = args.analysis / 'position_fallback_20261124.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'policy': 'Apply the trained rank-only one-exchange policy only where the fixed '
                   'external-six judge abstains. The candidate and score model are unchanged.',
                   'splits': reports}, stream, indent=1)
    print(json.dumps(reports, indent=1))


if __name__ == '__main__':
    main()
