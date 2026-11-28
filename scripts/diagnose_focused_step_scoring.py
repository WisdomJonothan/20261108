"""Probe focused ranking with the annotated missing step as a privileged input."""

import argparse
from collections import Counter
import json
from pathlib import Path

from eval_event_date_windows import write_gz
from replay_conditional_rank_fusion import key
from replay_contrastive_evidence_admission import read


def probe_rows(args):
    candidates = read(args.candidates)['per_query']
    prepared = {key(r): r for r in read(args.inputs)['per_query'] if r['view'] == 'gap'}
    domains = {r['domain'] for r in candidates}
    steps = {}
    for domain in domains:
        with (args.data_root/domain/'steps_train.jsonl').open(encoding='utf-8') as stream:
            steps[domain] = {r['id']: r['steps'] for line in stream for r in (json.loads(line),)}
    result = []
    for row in candidates:
        if key(row) not in prepared:
            continue
        base = prepared[key(row)]
        for step in steps[row['domain']].get(row['qid'], ()):
            evidence = set(step['gold_ids'])
            offered = [d for d in row['offers']['two_sided'] if d in evidence]
            if offered and evidence.isdisjoint(row['current']):
                result.append(base | {'query': step['step_instruction'], 'view': 'annotated_step',
                                      'step_id': step['step_id'], 'positive_offers': offered})
    return {'queries': len(result), 'per_query': result}


def diagnose(args):
    scores = read(args.scores)
    original = read(args.focused)
    baseline = {key(r): r for r in original['per_query'] if r['view'] == 'gap'}
    if scores['model'] != original['model']:
        raise ValueError('Expected the same text scorer for both requirements')
    counts = {mode: Counter(top=0, beats_tenth=0, beats_all_ten=0) for mode in ('gap', 'annotated_step')}
    results = []
    for probe in scores['per_query']:
        by_view = {}
        for mode, record in (('gap', baseline[key(probe)]), ('annotated_step', probe)):
            if record['current'] != probe['current'] or record['docs'] != probe['docs']:
                raise ValueError(f'Lists differ for {key(probe)}')
            values = dict(zip(record['docs'], record['new_scores'], strict=True))
            best = max(record['docs'][10:], key=values.__getitem__)
            qualifies = best in probe['positive_offers']
            outcomes = {'top': qualifies,
                        'beats_tenth': qualifies and values[best] > values[record['current'][9]],
                        'beats_all_ten': qualifies and values[best] > max(values[d] for d in record['current'])}
            counts[mode].update(outcomes)
            by_view[mode] = {'winner': best, **outcomes}
        results.append({name: probe[name] for name in ('domain', 'qid', 'step_id', 'positive_offers')}
                       | {'outcomes': by_view})
    report = {'matched_steps': len(results), 'queries': len({key(r) for r in results}),
              'counts': counts, 'per_step': results, 'annotated_scores': scores['per_query']}
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_step','annotated_scores')},
                     indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    first = stages.add_parser('prepare')
    for arg in ('candidates','inputs','data-root'):
        first.add_argument(f'--{arg}', type=Path, required=True)
    last = stages.add_parser('score')
    for arg in ('scores','focused'):
        last.add_argument(f'--{arg}', type=Path, required=True)
    for stage in (first,last):
        stage.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    write_gz(args.output, probe_rows(args) if args.stage == 'prepare' else diagnose(args))


if __name__ == '__main__':
    main()
