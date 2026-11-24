"""Count the annotated evidence steps gained and lost by each scored replacement."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from coverage_cf_bound import read_steps
from replay_position_discount import bootstrap


def coverage(top, steps):
    docs = set(top)
    return {step for step, gold in steps.items() if not docs.isdisjoint(gold)}


def compare(report, root, data_root, reference):
    folds = report['folds']
    split = 'dev' if folds == ['dev'] else 'train'
    suffixes = ('dev',) if split == 'dev' else tuple(f'train_{fold}' for fold in folds)
    manifests = {
        (item['domain'], item['qid']): item['frozen_top10']
        for source in suffixes
        for item in load(root / 'results/replacement_v2' / f'candidates_{source}.json')['queries']
    }
    steps = {domain: read_steps(data_root / domain / f'qrels_steps_{split}.txt')
             for domain in {row['domain'] for row in report['per_query']}}
    rows = []
    for row in report['per_query']:
        domain, qid = row['domain'], row['qid']
        if not (query_steps := steps[domain].get(qid)):
            continue
        top = manifests[domain, qid]
        before = coverage(top, query_steps)
        effects = {}
        for name, result in row['outcomes'].items():
            changed = top.copy()
            if action := result['choice']:
                slot = action['slot'] - 1
                if changed[slot] != action['incumbent'] or action['challenger'] in changed:
                    raise ValueError(f'Mismatched frozen ranking for {domain}/{qid}')
                changed[slot] = action['challenger']
            after = coverage(changed, query_steps)
            effects[name] = {'covered': len(after), 'gained': len(after - before), 'lost': len(before - after)}
        rows.append({'domain': domain, 'qid': qid, 'total_steps': len(query_steps),
                     'frozen_steps': len(before), 'arms': effects})
    arms = rows[0]['arms']
    summary = {'annotated_queries': len(rows), 'annotated_domains': len({r['domain'] for r in rows}),
               'frozen_coverage': bootstrap([(r['domain'], r['frozen_steps'] / r['total_steps']) for r in rows])['point'],
               'arms': {}}
    for name in arms:
        changes = [(r['domain'], (r['arms'][name]['covered'] - r['arms'][reference]['covered']) / r['total_steps'])
                   for r in rows]
        summary['arms'][name] = {
            'coverage': bootstrap([(r['domain'], r['arms'][name]['covered'] / r['total_steps']) for r in rows])['point'],
            'paired_vs_reference': bootstrap(changes),
            'steps_gained_vs_frozen': sum(r['arms'][name]['gained'] for r in rows),
            'steps_lost_vs_frozen': sum(r['arms'][name]['lost'] for r in rows),
        }
    return {'summary': summary, 'per_query': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--reference-arm', required=True)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root or args.root / 'data/track1_tempo'
    results = {path.name: compare(load(path), args.root, data_root, args.reference_arm) for path in args.reports}
    report = {'reference_arm': args.reference_arm, 'reports': results}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=1)
    print(json.dumps({key: value['summary'] for key, value in results.items()}, indent=1))


if __name__ == '__main__':
    main()
