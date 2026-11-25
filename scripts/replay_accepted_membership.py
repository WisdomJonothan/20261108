"""Replay different edit scopes on accepted and abstained six-pair queries."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index

ARMS = ('start', 'reorder_same_ten', 'one_fixed', 'one_reordered', 'unrestricted')


def evaluate(report):
    arms = report['starts']
    competition = index(arms['competition'])
    abstention = index(arms['abstention'])
    if competition.keys() != abstention.keys():
        raise ValueError('The starting lists must cover the same queries')
    rows = []
    for key, kept in competition.items():
        abstained = abstention[key]['active']
        if kept['rankings']['start'] != abstention[key]['rankings']['start']:
            raise ValueError(f'The two starting lists differ for {key}')
        scores = kept['scores']
        selected = 'unrestricted' if abstained else 'reorder_same_ten'
        if set(kept['rankings']['reorder_same_ten']) != set(kept['rankings']['start']):
            raise ValueError(f'Reordering changed list membership for {key}')
        rows.append({'domain': key[0], 'qid': key[1], 'abstained': abstained,
                     'active_arm': selected, 'scores': scores,
                     'routing_score': scores[selected],
                     'previous_fallback_score': abstention[key]['scores']['unrestricted'],
                     'routing_top10': kept['rankings'][selected]})

    def paired(more, less, subset=rows):
        return paired_delta([(r['domain'], more(r) - less(r)) for r in subset])

    def report_group(group):
        baseline = paired(lambda r: r['scores']['start'], lambda _: 0, group)['point']
        return {'queries': len(group), 'domains': len({r['domain'] for r in group}),
                'start': baseline,
                'change_by_edit': {arm: paired(lambda r: r['scores'][arm],
                                               lambda r: r['scores']['start'], group)
                                   for arm in ARMS if arm != 'start'}}

    base = paired(lambda r: r['scores']['start'], lambda _: 0)['point']
    delta = paired(lambda r: r['routing_score'], lambda r: r['scores']['start'])
    return {
        'queries': len(rows), 'domains': len({r['domain'] for r in rows}),
        'abstained': report_group([r for r in rows if r['abstained']]),
        'accepted': report_group([r for r in rows if not r['abstained']]),
        'routing_ndcg': base + delta['point'], 'versus_six_pair': delta,
        'versus_previous_fallback': paired(lambda r: r['routing_score'],
                                          lambda r: r['previous_fallback_score']),
        'versus_full8b': paired(lambda r: r['routing_score'], lambda r: r['scores']['unrestricted']),
        'per_query': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'protocol': 'Use fixed 8B scores to reorder accepted top tens and to rerank all 100 '
              'candidates following abstention. Compare edit scopes by query group and keep '
              'all ranking scores and starting lists from the matched archive.'}
    for split, path in (('oof', args.oof), ('dev', args.dev)):
        result = evaluate(load(path))
        report[split] = result
        print(split, result['routing_ndcg'], result['versus_previous_fallback']['point'],
              result['versus_previous_fallback']['ci95'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, separators=(',', ':'))
        stream.write('\n')


if __name__ == '__main__':
    main()
