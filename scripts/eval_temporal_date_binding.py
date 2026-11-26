"""Probe which event's date a reranker uses in matched document pairs."""

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
import re

from compare_anchor_reference import load
from eval_temporal_query_contrast import (
    SIDES, cross_encoder, evaluate, prepare, score_qwen_variants,
)


TOPICS = (
    ('Alder', 'Briar', 'inspection'),
    ('Cedar', 'Dover', 'installation'),
    ('Elm', 'Firth', 'calibration'),
    ('Grove', 'Haven', 'delivery'),
    ('Iris', 'Jasper', 'activation'),
    ('Kestrel', 'Larch', 'registration'),
)
YEARS = (
    (1888, 1901, 1912), (1939, 1945, 1951), (1971, 1974, 1982),
    (1994, 2000, 2008), (2012, 2015, 2017), (2019, 2020, 2021),
)
ARMS = ('both_dates', 'hide_other_date', 'hide_target_date')
RELATIONS = ('before', 'after')


def generate():
    cases = []
    inputs = {arm: [] for arm in ARMS}
    for first, second, event in TOPICS:
        names = (first, second)
        for early, cutoff, late in YEARS:
            dates = ((early, late), (late, early))
            family = f'{event}-{cutoff}'
            for order in ((0, 1), (1, 0)):
                trials = []
                for target, name in enumerate(names):
                    for relation, condition in enumerate(RELATIONS):
                        query = f'Which passage reports that the {name} site {event} occurred {condition} {cutoff}?'
                        for arm in ARMS:
                            hide = {'both_dates': None, 'hide_other_date': 1 - target,
                                    'hide_target_date': target}[arm]
                            passages = [
                                ' '.join(f'The {names[i]} site {event} occurred in '
                                         f'{"[year withheld]" if i == hide else assigned[i]}.'
                                         for i in order)
                                for assigned in dates
                            ]
                            if arm == 'both_dates':
                                counts = [Counter(re.findall(r'\w+', d)) for d in passages]
                                if counts[0] != counts[1] or passages[0] == passages[1]:
                                    raise ValueError(f'Unmatched test passages for {family}')
                            inputs[arm].extend((query, passage) for passage in passages)
                        trials.append({'target': name, 'relation': condition, 'gold': target ^ relation})
                cases.append({'family': family, 'first_subject': names[order[0]], 'trials': trials})
    return cases, inputs


def summarize_binding(cases, scores):
    if len(scores) != 8 * len(cases) or not all(isfinite(s) for s in scores):
        raise ValueError('Every prompt must have a finite score')
    rows = []
    successful_orders = Counter()
    for i, case in enumerate(cases):
        trials = []
        for j, trial in enumerate(case['trials']):
            offset = 8 * i + 2 * j
            pair = scores[offset:offset + 2]
            gold = trial['gold']
            trials.append({**trial, 'scores': pair, 'margin': pair[gold] - pair[1 - gold]})
        if all(trial['margin'] > 0 for trial in trials):
            successful_orders[case['family']] += 1
        rows.append({**case, 'trials': trials})
    return {
        'correct_preferences': sum(t['margin'] > 0 for row in rows for t in row['trials']),
        'tied_preferences': sum(t['margin'] == 0 for row in rows for t in row['trials']),
        'preferences': 4 * len(rows),
        'complete_orders': sum(successful_orders.values()),
        'orders': len(rows),
        'complete_families': sum(count == 2 for count in successful_orders.values()),
        'families': len({row['family'] for row in rows}),
        'per_case': rows,
    }


def build_source_inputs(examples, annotations):
    variants = {arm: [] for arm in ('source', 'source_aligned_note', 'source_exchanged_note')}
    for case, queries, docs in examples:
        dates = annotations[case['id']]
        for query in (queries[side] for side in SIDES):
            for side, other in (('early', 'late'), ('late', 'early')):
                variants['source'].append((query, docs[side]))
                for arm, date in (('source_aligned_note', dates[side]),
                                  ('source_exchanged_note', dates[other])):
                    variants[arm].append((query, docs[side] + f'\nUnrelated date note: {date}.'))
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('qwen', 'bge'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--precision', choices=('float32', 'bfloat16'), default='float32')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    cases, inputs = generate()
    examples = prepare(load(args.manifest)['cases'], args.data_root)
    inputs.update(build_source_inputs(examples, load(args.dates)['times']))
    options = {'device': args.device, 'precision': args.precision}
    instruction = None
    if args.kind == 'qwen':
        from run_qwen3_rerank import INSTRUCT

        instruction = INSTRUCT
        raw = score_qwen_variants(args.model, inputs, dict.fromkeys(inputs, instruction), 4, **options)
    else:
        lengths = [len(pairs) for pairs in inputs.values()]
        flat = [pair for pairs in inputs.values() for pair in pairs]
        values, seconds = cross_encoder(args.model, flat, 4, **options)
        raw = {}
        offset = 0
        for arm, length in zip(inputs, lengths):
            raw[arm] = (values[offset:offset + length], None)
            offset += length
    results = {}
    for arm, (scores, duration) in raw.items():
        if len(scores) != len(inputs[arm]) or not all(isfinite(s) for s in scores):
            raise ValueError(f'Missing or non-finite scores in {arm}')
        if arm in ARMS:
            result = summarize_binding(cases, scores)
        else:
            result = evaluate(arm, [scores[i:i + 4] for i in range(0, len(scores), 4)], examples)
        results[arm] = result
        if duration is not None:
            result['scoring_seconds'] = duration
        print(arm, {key: value for key, value in result.items() if key != 'per_case'}, flush=True)
    report = {'kind': args.kind, 'model': args.model.name, 'instruction': instruction,
              **options, 'scoring_seconds': seconds if args.kind == 'bge' else sum(v[1] for v in raw.values()),
              'source_manifest': args.manifest.name, 'source_dates': args.dates.name, 'results': results}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
