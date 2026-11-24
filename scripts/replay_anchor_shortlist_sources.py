"""Compare deterministic candidate selection rules using the same cached judge."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import ordered_scores, paired_delta
from replay_anchor_fallback import index


def select_candidates(top, ordered_frozen):
    protected = set(top)
    dense = [doc for doc, _ in ordered_frozen if doc not in protected]
    fusion = [doc for doc, _ in sorted(ordered_frozen, key=lambda pair: -pair[1])
              if doc not in protected]
    mixed = list(dict.fromkeys(dense[:5] + fusion))[:10]
    return {'dense': protected | set(dense[:10]), 'fused': protected | set(fusion[:10]),
            'mixed': protected | set(mixed)}


def evaluate(inputs, base_cache, data, split):
    rows = []
    labels = {domain: read_qrels(data / domain / f'qrels_{split}.txt')
              for domain in sorted({domain for r in inputs for domain, _ in r})}
    for cohort, report in enumerate(inputs, 1):
        for key, row in report.items():
            domain, qid = key
            base = base_cache[key]['frozen_scores']
            scores = row['scores']
            gold = labels[domain][qid]
            top = row['arms']['base']['ranking']
            if len(base) != 100 or {doc for doc, _ in base} != scores.keys():
                raise ValueError(f'Score cache differs from the original candidate set for {key}')
            arms = select_candidates(top, base)
            if any(len(members) != 20 for members in arms.values()):
                raise ValueError(f'Every shortlist must have twenty documents for {key}')
            rankings = {name: ordered_scores([(doc, value) for doc, value in base if doc in members],
                                             scores)[:10] for name, members in arms.items()}
            rankings['full'] = ordered_scores(base, scores)[:10]
            if rankings['full'] != row['arms']['base_plus_anchor30']['ranking']:
                raise ValueError(f'Full-pool reconstruction differs for {key}')
            arms['full'] = set(scores)
            rows.append({
                'domain': domain, 'qid': qid, 'cohort': cohort, 'rankings': rankings,
                'scores': {name: ndcg10(rank, gold) for name, rank in rankings.items()},
                'oracle': {name: ndcg10(sorted(gold & members), gold) for name, members in arms.items()},
                'judged_relevant_outside_start': {name: len((members - set(top)) & gold)
                                                  for name, members in arms.items()},
                'members_changed_vs_dense': {name: len(members - arms['dense']) for name, members in arms.items()},
            })
    return rows


def summarize(rows):
    def macro(values):
        domains = defaultdict(list)
        for row, value in zip(rows, values):
            domains[row['domain']].append(value)
        return mean(map(mean, domains.values()))

    names = tuple(rows[0]['scores'])
    return {
        'queries': len(rows), 'domains': len({r['domain'] for r in rows}),
        'scores': {name: macro([row['scores'][name] for row in rows]) for name in names},
        'oracle': {name: macro([row['oracle'][name] for row in rows]) for name in names},
        'delta_vs_dense': {name: paired_delta([(r['domain'], r['scores'][name] - r['scores']['dense'])
                                               for r in rows])
                           for name in names if name != 'dense'},
        'queries_with_additional_positive': {name: sum(r['judged_relevant_outside_start'][name] > 0 for r in rows)
                                               for name in names},
        'mean_members_changed_vs_dense': {name: mean(r['members_changed_vs_dense'][name] for r in rows)
                                         for name in names},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pool-scores', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [load(path) for path in args.inputs]
    split = reports[0]['split']
    if any(r['split'] != split or r['pool_depth'] != 100 for r in reports):
        raise ValueError('Expected score caches at depth 100 from a single split')
    indexed = [index(report) for report in reports]
    if len(set().union(*(r.keys() for r in indexed))) != sum(map(len, indexed)):
        raise ValueError('Evaluation cohorts must be disjoint')
    rows = evaluate(indexed, index(load(args.pool_scores)), args.data_root, split)
    result = {
        'protocol': 'Preserve the original ten and compare three fixed ways of selecting ten outsiders. '
                    'Use ten highest dense candidates, ten highest frozen-fusion candidates, or five highest '
                    'dense candidates then the five highest distinct frozen-fusion candidates. '
                    'Recompute both z-scores in each shortlist. Qrels score rankings and oracle ceilings only.',
        'split': split, 'inputs': [path.name for path in args.inputs], 'pooled': summarize(rows),
        'by_input': {path.name: summarize([r for r in rows if r['cohort'] == i])
                     for i, path in enumerate(args.inputs, 1)},
        'per_query': rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, separators=(',', ':'))
        stream.write('\n')
    for cohort, summary in [('pooled', result['pooled']), *result['by_input'].items()]:
        print(cohort, summary['scores'], summary['oracle'])
    print({k: {x: v[x] for x in ('point', 'ci95', 'domain_bootstrap_ci95')}
           for k, v in result['pooled']['delta_vs_dense'].items()})


if __name__ == '__main__':
    main()
