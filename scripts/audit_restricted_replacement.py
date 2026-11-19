"""Oracle gains from replacing fixed top10 slots with disagreement candidates."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_run(path):
    queries = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        qid, _, doc, rank, _, _ = line.split()
        queries.setdefault(qid, []).append((int(rank), doc))
    result = {q: [doc for _, doc in sorted(rows)] for q, rows in queries.items()}
    assert all(len(v) == len(set(v)) for v in result.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    experiment = root
    domains = sorted(json.loads((experiment / 'results/fusion100_full_ce_s17_20261115_dev_eval.json').read_text(encoding='utf-8'))['domains'])
    discounts = np.array([1 / math.log2(i + 2) for i in range(10)])
    records = []
    for domain in domains:
        base = read_run(root / 'runs' / f'sub_sft_e2bc_{domain}_dev_head10.trec')
        dense = read_run(root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        ce = read_run(experiment / 'runs' / f'fusion100_full_ce_s17_20261115_{domain}_dev_ce.trec')
        query_ids = {json.loads(line)['id'] for line in (root / 'data/track1_tempo' / domain / 'examples_dev.jsonl').read_text(encoding='utf-8').splitlines()}
        gold = {q: set() for q in query_ids}
        for line in (root / 'data/track1_tempo' / domain / 'qrels_dev.txt').read_text(encoding='utf-8').splitlines():
            q, _, doc, rel = line.split()
            if int(rel) > 0:
                gold[q].add(doc)
        assert query_ids == set(base) == set(dense) == set(ce)
        for q in sorted(query_ids):
            top = base[q][:10]
            assert len(top) == 10
            relevant = gold[q]
            ideal = float(discounts[:min(10, len(relevant))].sum())
            assert ideal > 0
            current = sum(discounts[i] for i, doc in enumerate(top) if doc in relevant) / ideal
            # The highest-weight nonrelevant slots are the optimal replacements
            # for binary qrels, with every other slot frozen.
            available_gains = [discounts[i] / ideal for i, doc in enumerate(top) if doc not in relevant]
            for depth in [10, 20, 100]:
                candidates = (set(dense[q][:depth]) | set(ce[q][:depth])) - set(top)
                incoming = len(candidates & relevant)
                scores = {str(k): current + sum(available_gains[:min(k, incoming)]) for k in [1, 2, 3]}
                pool_positives = len((set(top) | candidates) & relevant)
                rerank_oracle = float(discounts[:min(10, pool_positives)].sum()) / ideal
                records.append({'domain': domain, 'qid': q, 'candidate_depth_per_source': depth,
                                'baseline': float(current), 'candidate_count': len(candidates),
                                'relevant_incoming_count': incoming,
                                'useful_slot_count': min(incoming, len(available_gains)),
                                'oracle_fixed_slots': scores,
                                'unrestricted_rerank_same_union_oracle': rerank_oracle})
    assert len(domains) == 13 and len(records) == 519 * 3
    summary = {}
    for depth in [10, 20, 100]:
        per_domain = {}
        for domain in domains:
            rows = [r for r in records if r['domain'] == domain and r['candidate_depth_per_source'] == depth]
            per_domain[domain] = {'n': len(rows), 'baseline': float(np.mean([r['baseline'] for r in rows])),
                                  'oracle': {str(k): float(np.mean([r['oracle_fixed_slots'][str(k)] for r in rows])) for k in [1, 2, 3]},
                                  'unrestricted_rerank_same_union_oracle': float(np.mean([r['unrestricted_rerank_same_union_oracle'] for r in rows])),
                                  'queries_with_useful_replacement': sum(r['useful_slot_count'] > 0 for r in rows)}
        summary[str(depth)] = {'baseline_macro': float(np.mean([v['baseline'] for v in per_domain.values()])),
                               'oracle_macro': {str(k): float(np.mean([v['oracle'][str(k)] for v in per_domain.values()])) for k in [1, 2, 3]},
                               'unrestricted_rerank_same_union_oracle_macro': float(np.mean([v['unrestricted_rerank_same_union_oracle'] for v in per_domain.values()])),
                               'queries_with_useful_replacement': sum(v['queries_with_useful_replacement'] for v in per_domain.values()),
                               'per_domain': per_domain}
    assert abs(summary['10']['baseline_macro'] - 0.3580985) < 1e-6
    output = {'protocol': 'Exploratory dev oracle diagnosis. Frozen submission sub_sft_e2bc top10. Incoming candidates are the union of dense and independently trained full-question CE top-depth, excluding current top10. Binary qrels select ideal replacements only for upper-bound calculation. Replace up to k existing slots; other positions and ordering stay fixed. No relevant documents removed. Full query gold determines ideal DCG. This assumes a perfect replacement decision and identifies no executable policy.',
              'queries': 519, 'domains': 13, 'summary': summary, 'per_query': records}
    path = root / 'results/restricted_replacement_oracle.json'
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps({d: {k: v for k, v in s.items() if k != 'per_domain'} for d, s in summary.items()}, indent=2))


if __name__ == '__main__':
    main()
