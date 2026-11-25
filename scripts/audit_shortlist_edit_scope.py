"""Measure how edit position constrains the same six challenger documents."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels


def evaluate(rows, data):
    gold = {d: read_qrels(data / d / 'qrels_dev.txt') for d in {r['domain'] for r in rows}}
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        domain = row['domain']
        start = row['start']
        relevant = gold[domain][row['qid']]
        arms = row['shortlists']

        def score(documents):
            return ndcg10(documents, relevant)

        def best_order(documents):
            return score(sorted(documents, key=lambda d: d not in relevant))

        baseline = score(start)
        same_ten = best_order(start)
        values['baseline'][domain].append(baseline)
        values['same_ten_reordered'][domain].append(same_ten)
        for arm, docs in arms.items():
            if len(start) != 10 or len(set(start)) != 10 or set(start).intersection(docs):
                raise ValueError(f'Invalid start or challengers for {domain}/{row["qid"]}')
            variants = [start[:i] + [d] + start[i + 1:] for d in docs for i in range(10)]
            values[arm + '_slot10'][domain].append(max(baseline, *(score(v) for v in variants[9::10])))
            values[arm + '_any_slot'][domain].append(max(baseline, *(score(v) for v in variants)))
            values[arm + '_one_then_reorder'][domain].append(
                max(same_ten, *(best_order(v) for v in variants)))
        union = list(set().union(*arms.values()))
        values['unrestricted_shortlist_union'][domain].append(best_order(start + union))
    return {name: mean(map(mean, by_domain.values())) for name, by_domain in values.items()}


def width_sweep(rows, data, runs_root):
    """Compare how to spend the same number of extra candidate slots."""
    from lib_reteco import read_trec_run

    widths = (6, 12, 24, 48, 72)
    sources = ('dense_mxbai', 'dense_e5base', 'dense_bge')
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row['domain']].append(row)
    totals = defaultdict(dict)
    for domain, domain_rows in by_domain.items():
        gold = read_qrels(data / domain / 'qrels_dev.txt')
        runs = [read_trec_run(runs_root / f'{source}_{domain}_dev.trec') for source in sources]
        results = defaultdict(list)
        for row in domain_rows:
            qid, start = row['qid'], row['start']
            relevant = gold[qid]
            rankings = [[doc for _, _, doc in run[qid][:300]] for run in runs]
            dense = rankings[0]
            rrf = defaultdict(float)
            for ranking in rankings:
                for rank, doc in enumerate(dict.fromkeys(ranking), 1):
                    rrf[doc] += 1 / (60 + rank)

            def order(documents):
                return sorted(documents, key=lambda doc: (-rrf[doc], doc))

            inside = order(set(dense[:100]) - set(start))
            tail = dense[100:]
            others = order((set(rankings[1]) | set(rankings[2])) - set(dense))
            mixed = list(dict.fromkeys(doc for group in zip(order(tail), tail, others) for doc in group))
            for width in widths:
                half = width // 2
                pools = {
                    'inside_only': inside[:width],
                    'dense_expand': inside[:half] + tail[:half],
                    'mixed_expand': inside[:half] + mixed[:half],
                }
                for name, docs in pools.items():
                    if len(set(docs)) != width:
                        raise ValueError(f'Budget mismatch for {domain}/{qid}/{name}')
                    ranked = sorted(start + docs, key=lambda doc: doc not in relevant)
                    results[width, name].append(ndcg10(ranked, relevant))
        for key, scores in results.items():
            totals[key][domain] = mean(scores)
    return {str(width): {name: mean(totals[width, name].values())
                         for name in ('inside_only', 'dense_expand', 'mixed_expand')}
            for width in widths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data/track1_tempo'))
    parser.add_argument('--runs-root', type=Path, default=Path('runs'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.manifest)['queries']
    if len(rows) != 519 or len({r['domain'] for r in rows}) != 13:
        raise ValueError('Expected every official-dev query')
    report = {
        'protocol': 'Use qrels to optimize the same six challengers under rank-ten, any single slot, '
                    'and single-admission plus reorder constraints. Also measure the label-based '
                    'best ordering of the starting ten and of the full shortlist union.',
        'queries': len(rows), 'domain_macro': evaluate(rows, args.data_root),
        'width_sweep_oracle': width_sweep(rows, args.data_root, args.runs_root),
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
