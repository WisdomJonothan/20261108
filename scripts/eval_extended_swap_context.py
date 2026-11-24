"""Evaluate symmetric excerpt expansion on every routed suffix pair in a fold."""
import argparse
import json
from pathlib import Path
from statistics import mean

from compare_anchor_reference import load
from eval_abstention_shortlist import outcome
from eval_joint_swap_support import prepare, summarize
from eval_swap_context_extent import build_cases, score
from replay_position_discount import eligible


POLICIES = ('external6', 'direct', 'both_long')


def pair_key(row):
    return row['domain'], row['qid'], row['incumbent'], row['challenger']


def score_changed(cases, checkpoint, cached):
    pending = []
    counts = {'unchanged': 0, 'cached': 0, 'new': 0}
    for case in cases:
        if case['variants']['both_long'] == case['variants']['short']:
            counts['unchanged'] += 1
        elif prior := cached.get(pair_key(case)):
            lengths = {side: len(text.split()) for side, text in case['variants']['both_long'].items()}
            if (prior['words']['both_long'] != lengths
                    or prior['historical_probabilities'] != case['historical_probabilities']):
                raise ValueError(f'Context cache differs for {pair_key(case)}')
            case['predictions']['both_long'] = prior['predictions']['both_long']
            counts['cached'] += 1
        else:
            pending.append(case)
            counts['new'] += 1
    stats = score(pending, checkpoint, variants=('both_long',))['both_long'] if pending else {}
    if stats.get('prompts', 0) != 2 * counts['new']:
        raise ValueError('Unexpected fresh prompt count')
    return stats, counts


def evaluate(rows, cases):
    matched = {pair_key(case): case for case in cases}
    saved = []
    for row in rows:
        direct = row['arms']['supervised']
        values = {'external6': outcome(row['arms']['external6'], True), 'direct': outcome(direct, True)}
        expanded = []
        diagnostics = []
        for action, label in zip(direct[3:], row['treatment']):
            key = (row['domain'], row['qid'], action['incumbent'], action['challenger'])
            pair = matched.pop(key)
            new = action.copy()
            pred = pair['predictions'].get('both_long')
            if pred:
                new['p_fwd'] = pred['vote']['fwd']['challenger']
                new['p_rev'] = pred['vote']['rev']['challenger']
                new['mp'] = mean((new['p_fwd'], new['p_rev']))
                new['agree'] = all(max(vote, key=vote.get) == 'challenger' for vote in pred['vote'].values())
                if bool(eligible([new])) != pred['accepted']:
                    raise ValueError(f'Expanded decision does not reproduce for {key}')
            expanded.append(new)
            diagnostics.append(label | {
                'new_probability': new['mp'], 'expanded_vote': pred['vote'] if pred else None,
                'accepted': {'direct': bool(eligible([action])), 'both_long': bool(eligible([new]))},
                'words': {name: {side: len(text.split()) for side, text in variant.items()}
                          for name, variant in pair['variants'].items() if name in ('short', 'both_long')},
            })
        values['both_long'] = outcome(direct[:3] + expanded if expanded else direct, True)
        saved.append({'domain': row['domain'], 'qid': row['qid'], 'fold': row['fold'], 'baseline': row['baseline'],
                      'routed': row['routed'], 'outcomes': values, 'support_judgments': diagnostics})
    if matched:
        raise ValueError('Some expanded comparisons do not match the fixed actions')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--direct-report', type=Path, required=True)
    parser.add_argument('--pair-cache', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    data = args.data_root or args.root / 'data/track1_tempo'
    reference = load(args.direct_report)
    cached = {}
    if args.pair_cache:
        report = load(args.pair_cache)
        if report['folds'] != reference['folds']:
            parser.error('Pair cache must come from the same folds as the Direct reference.')
        cached = {pair_key(pair): pair for pair in report['pairs']}
    cases = build_cases(args.root, data, reference, exclusive_only=False)
    stats, counts = score_changed(cases, args.model, cached)
    rows = evaluate(prepare(args.root, data, reference, reference['folds']), cases)
    result = {
        'protocol': 'Keep all six supervised-shortlist comparison slots and the Direct prompt. '
                    'Expand both sides to at most 384 words in the routed suffix. Use the same '
                    'two-order gate and one replacement at most. Score all routed pairs. '
                    'Unchanged text uses its historical scores, and valid diagnostic pairs reuse '
                    'their expanded scores. Qrels enter evaluation only.',
        'folds': reference['folds'], 'scored_pairs': counts, 'inference': stats,
        'summary': summarize(rows, POLICIES), 'per_query': rows,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps({'scored_pairs': counts, 'summary': result['summary']}, indent=1), flush=True)


if __name__ == '__main__':
    main()
