"""Join fixed official-dev rankings and audit relevance and evidence coverage."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels, read_steps
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import discounted_changes, index


def macro(rows):
    domains = defaultdict(list)
    for domain, value in rows:
        domains[domain].append(value)
    return mean(map(mean, domains.values()))


def coverage(ranking, steps):
    documents = set(ranking)
    return mean(bool(documents & step) for step in steps.values())


def prepare(raw, full, capped, control, data_root):
    main = index(raw)
    competition = index(full['starts']['competition'])
    fallback = index(full['starts']['abstention'])
    cap = index(capped['starts']['frozen'])
    direct = index(control['starts']['frozen'])
    if any(source.keys() != main.keys() for source in (competition, fallback, cap, direct)):
        raise ValueError('Every policy must cover the same official dev queries')
    if len(main) != 519 or len({d for d, _ in main}) != 13:
        raise ValueError('Expected all 519 official dev queries in 13 domains')

    labels = {d: (read_qrels(data_root / d / 'qrels_dev.txt'),
                  read_steps(data_root / d / 'qrels_steps_dev.txt'))
              for d in sorted({d for d, _ in main})}
    rows = []
    for (domain, qid), item in main.items():
        key = domain, qid
        gold = labels[domain][0][qid]
        ed = competition[key]
        rankings = {
            'frozen': item['arms']['base']['ranking'],
            'six_pair': ed['rankings']['start'],
            'qwen8b_100_only': item['arms']['anchor_only']['ranking'],
            'qwen8b_100_fusion': ed['rankings']['unrestricted'],
            'qwen8b_100_same_ten': ed['rankings']['reorder_same_ten'],
            'qwen8b_100_one_fixed': ed['rankings']['one_fixed'],
            'qwen8b_100_one_reordered': ed['rankings']['one_reordered'],
            'qwen8b_100_abstention': fallback[key]['rankings']['unrestricted'],
            'qwen8b_20_replay': cap[key]['rankings']['unrestricted'],
            'qwen4b_20_direct': direct[key]['rankings']['unrestricted'],
        }
        if any(len(ranking) != 10 or len(set(ranking)) != 10 for ranking in rankings.values()):
            raise ValueError(f'Expected ten unique documents in each ranking for {key}')
        if rankings['qwen8b_100_fusion'] != item['arms']['base_plus_anchor30']['ranking']:
            raise ValueError(f'Replayed fusion disagrees with the raw score report for {key}')
        scores = {name: ndcg10(ranking, gold) for name, ranking in rankings.items()}
        for score, expected in (
            (scores['frozen'], item['arms']['base']['ndcg']),
            (scores['six_pair'], ed['external6']),
            (scores['qwen8b_100_fusion'], item['arms']['base_plus_anchor30']['ndcg']),
            (scores['qwen8b_20_replay'], cap[key]['scores']['unrestricted']),
            (scores['qwen4b_20_direct'], direct[key]['scores']['unrestricted']),
        ):
            if abs(score - expected) > 1e-12:
                raise ValueError(f'Cached scores do not match official qrels for {key}')
        primary = rankings['qwen8b_100_fusion']
        start = rankings['six_pair']
        effects = discounted_changes(start, primary, gold)
        if abs(sum(effects['rank_delta']) - (scores['qwen8b_100_fusion'] - scores['six_pair'])) > 1e-12:
            raise ValueError(f'Rank changes do not add to the primary effect for {key}')
        row = {
            'domain': domain, 'qid': qid, 'scores': scores, 'has_positive_at_100': ed['has_pool_positive'],
            'abstained': fallback[key]['active'], 'discounted_effects': effects,
            'primary_relevant_arrivals': len((set(primary) - set(start)) & gold),
            'primary_relevant_departures': len((set(start) - set(primary)) & gold),
        }
        if steps := labels[domain][1].get(qid):
            row['step_coverage'] = {name: coverage(ranking, steps) for name, ranking in rankings.items()}
        rows.append(row)
    return rows


def summarize(rows):
    arms = rows[0]['scores']
    zeroes = [r for r in rows if r['scores']['six_pair'] == 0.0]
    annotated = [r for r in rows if 'step_coverage' in r]
    paired = {}
    for a, b in (
        ('qwen8b_100_fusion', 'six_pair'), ('qwen8b_100_fusion', 'frozen'),
        ('qwen8b_100_fusion', 'qwen8b_20_replay'), ('qwen8b_20_replay', 'qwen4b_20_direct'),
        ('qwen8b_100_one_reordered', 'qwen8b_100_one_fixed'),
        ('qwen8b_100_fusion', 'qwen8b_100_one_reordered'),
        ('qwen8b_100_abstention', 'six_pair'),
    ):
        paired[f'{a}_vs_{b}'] = paired_delta([(r['domain'], r['scores'][a] - r['scores'][b]) for r in rows])
    return {
        'queries': len(rows), 'domains': len({r['domain'] for r in rows}),
        'ndcg': {arm: macro((r['domain'], r['scores'][arm]) for r in rows) for arm in arms},
        'paired_deltas': paired,
        'six_pair_or_qwen8b_oracle': macro(
            (r['domain'], max(r['scores']['six_pair'], r['scores']['qwen8b_100_fusion'])) for r in rows),
        'competition_zeroes': len(zeroes),
        'competition_zeroes_retrievable_at_100': sum(r['has_positive_at_100'] for r in zeroes),
        'recovered_zeroes': {arm: sum(r['scores'][arm] > 0.0 for r in zeroes) for arm in arms},
        'new_zeroes': {arm: sum(r['scores'][arm] == 0.0 < r['scores']['six_pair'] for r in rows) for arm in arms},
        'step_coverage': {arm: macro((r['domain'], r['step_coverage'][arm]) for r in annotated) for arm in arms},
        'step_coverage_queries': len(annotated),
        'primary_discounted_effects': {
            **{field: macro((r['domain'], r['discounted_effects'][field]) for r in rows)
               for field in ('added_documents', 'removed_documents', 'retained_documents')},
            'by_rank': [macro((r['domain'], r['discounted_effects']['rank_delta'][i]) for r in rows)
                        for i in range(10)],
            'relevant_arrivals': sum(r['primary_relevant_arrivals'] for r in rows),
            'relevant_departures': sum(r['primary_relevant_departures'] for r in rows),
        },
        'abstentions': sum(r['abstained'] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--replay20', type=Path, required=True)
    parser.add_argument('--control20', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, help='Verified effective batch sizes for a sensitivity check.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw, full, cap, control = [load(path) for path in (args.raw, args.full, args.replay20, args.control20)]
    if any(source['split'] != 'dev' for source in (raw, full, cap, control)):
        raise ValueError('Expected official dev inputs')
    if [item['pool_depth'] for item in (raw, full, cap, control)] != [100, 100, 20, 20]:
        raise ValueError('Unexpected candidate depths')
    rows = prepare(raw, full, cap, control, args.data_root)
    summary = summarize(rows)
    if args.recovery:
        recovery = load(args.recovery)
        if recovery['queries'] != len(rows) or recovery['prompts'] != raw['inference']['prompts']:
            raise ValueError('Recovery summary differs from the official dev run')
        retry_ids = {(r['domain'], r['qid']) for r in recovery['retry_queries']}
        if not retry_ids <= {(r['domain'], r['qid']) for r in rows}:
            raise ValueError('Unknown IDs in recovery summary')
        regular = [r for r in rows if (r['domain'], r['qid']) not in retry_ids]
        summary['nominal_batches_only'] = {
            'queries': len(regular), 'domains': len({r['domain'] for r in regular}),
            'paired_primary_delta': paired_delta([
                (r['domain'], r['scores']['qwen8b_100_fusion'] - r['scores']['six_pair']) for r in regular
            ]),
        }
    report = {
        'protocol': 'Match the six-pair, Qwen3-8B depth-100 and depth-20 score replays, '
                    'and the earlier direct Qwen3-4B depth-20 results. Measure evidence '
                    'coverage on queries with positive step labels.',
        'inference': raw['inference'], 'summary': summary, 'per_query': rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
