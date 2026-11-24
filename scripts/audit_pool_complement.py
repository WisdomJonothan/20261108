"""Test a fixed 100-document pool with a protected output and companion retrievers."""
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from coverage_cf_bound import DOMAINS, ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_position_discount import bootstrap


def ordered_ids(run, qid, limit):
    return [doc for _, _, doc in run[qid][:limit]]


def summarize(rows):
    report = {'queries': len(rows)}
    for metric in ('recall', 'ndcg_oracle', 'hit'):
        differences = [(r['domain'], r['alternative'][metric] - r['original'][metric]) for r in rows]
        report[metric] = {key: bootstrap([(r['domain'], r[key][metric]) for r in rows])['point']
                          for key in ('original', 'alternative')}
        report[metric]['paired_delta'] = bootstrap(differences)
    report['gain_hit'] = sum(not r['original']['hit'] and bool(r['alternative']['hit']) for r in rows)
    report['lose_hit'] = sum(bool(r['original']['hit']) and not r['alternative']['hit'] for r in rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    trace = json.loads((root / 'results/analysis/position_oracle_dev_20261124.json')
                       .read_text(encoding='utf-8'))['per_query']
    decisions = {(r['domain'], r['qid']): r['policies']['external6']['chosen'] for r in trace}
    domains = defaultdict(list)
    for domain in DOMAINS:
        dense, bge, e5 = [
            read_trec_run(root / 'runs' / f'{tag}_{domain}_dev.trec')
            for tag in ('dense_mxbai', 'dense_bge', 'dense_e5base')]
        saved = read_trec_run(root / 'runs' / f'sub_sft_e2bc_{domain}_dev_head10.trec')
        gold = read_qrels(args.data_root / domain / 'qrels_dev.txt')
        assert set(dense) == set(bge) == set(e5) == set(saved) == set(gold)
        for qid in sorted(dense):
            original = ordered_ids(dense, qid, 100)
            top = ordered_ids(saved, qid, 10)
            chosen = decisions[domain, qid]
            if chosen:
                assert chosen['incumbent'] == top[chosen['slot'] - 1]
                top[chosen['slot'] - 1] = chosen['challenger']
            assert len(set(original)) == 100 and set(top) <= set(original)
            protected = list(dict.fromkeys(top + original))[:80]
            outside = []
            for docs in itertools.zip_longest(ordered_ids(bge, qid, 100), ordered_ids(e5, qid, 100)):
                outside.extend(doc for doc in docs if doc is not None)
            alternative = list(dict.fromkeys(protected + outside + original))[:100]
            assert len(alternative) == 100 and set(top) <= set(alternative)
            relevant = gold[qid]
            ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
            row = {'domain': domain, 'qid': qid, 'current_zero': ndcg10(top, relevant) <= 1e-12,
                   'new_docs': len(set(alternative) - set(original))}
            for key, pool in [('original', original), ('alternative', alternative)]:
                positives = len(set(pool) & relevant)
                dcg = sum(1 / math.log2(i + 2) for i in range(min(10, positives)))
                row[key] = {'recall': positives / len(relevant), 'ndcg_oracle': dcg / ideal,
                            'hit': int(positives > 0)}
            domains[domain].append(row)
    rows = [r for group in domains.values() for r in group]
    assert len(rows) == 519 and sum(r['current_zero'] for r in rows) == 130
    zero = [r for r in rows if r['current_zero']]
    report = {'protocol': 'Keep all ten output documents of the frozen competition system, fill '
              'to 80 with original mxbai, then fill to 100 alternating BGE and E5 runs at top100, '
              'with mxbai fallback if needed. No qrels in selection. Oracle and recall only.',
              'summary': summarize(rows), 'remaining_zero_queries': summarize(zero),
              'per_query': rows}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps({k: report[k] for k in ('summary', 'remaining_zero_queries')}, indent=1))


if __name__ == '__main__':
    main()
