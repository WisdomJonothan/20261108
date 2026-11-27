"""Evaluate the matched MiniLM objectives with a locked transfer and official replay."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from eval_event_date_windows import write_gz
from eval_fifth_ranker import order_inside
from replay_conditional_rank_fusion import key, labels, measure
from replay_incumbent_evidence import effect, macro


TRAINED = ('independent', 'fusion')


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def read_scores(path, panel):
    report = load(path)
    rows = {key(r): r for r in report['per_query'] if r['panel'] == panel}
    return report['model'], rows


def evaluate(args):
    if args.stage in ('screen', 'adapter-screen'):
        expected, panel = 242, 'reserve-242'
        controls = {'frozen': args.frozen, 'fusion': args.fusion}
        if args.stage == 'screen':
            controls['independent'] = args.independent
        selected = None
    else:
        locked = load(args.selection)
        expected_lock = 242 if args.stage == 'confirm' else 191
        if not locked['eligible'] or locked['queries'] != expected_lock or locked['selected'] not in TRAINED:
            raise ValueError('Expected a passing result from the previous stage')
        selected = locked['selected']
        if args.stage == 'confirm':
            expected, panel = 191, 'reserve-191'
            controls = {'frozen': args.frozen, selected: args.scores}
        else:
            expected, panel = 519, 'official-519'
            controls = {selected: args.scores}
    models, sources = {}, {}
    for arm, path in controls.items():
        models[arm], sources[arm] = read_scores(path, panel)
    if selected and models[selected] != locked['models'][selected]:
        raise ValueError('The held-out stage must use the selected checkpoint')
    first = next(iter(sources.values()))
    if len(first) != expected or any(rows.keys() != first.keys() for rows in sources.values()):
        raise ValueError('Score caches must cover the same complete evaluation population')
    if selected and first.keys() & {key(r) for r in locked['per_query']}:
        raise ValueError('Screening and evaluation populations overlap')
    rows = []
    for question, base in first.items():
        current = base['current']
        if len(current) != 10 or len(set(current)) != 10:
            raise ValueError(f'Invalid maintained rank for {question}')
        orders = {'current': current}
        for arm, model_outputs in sources.items():
            row = model_outputs[question]
            if row['current'] != current or row['prior_score'] != base['prior_score']:
                raise ValueError(f'Mismatched reference for {question}')
            raw = dict(zip(row['docs'], row['new_scores'], strict=True))
            orders[arm] = order_inside(current, raw)
            if set(orders[arm]) != set(current) or orders[arm][0] != current[0]:
                raise ValueError(f'New rank changed the first position or membership for {question}')
        rows.append({k: base[k] for k in ('domain', 'qid', 'panel', 'prior_score')} | {'rankings': orders})
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    rows = measure(rows, labels(rows, args.data_root, 'dev' if args.stage == 'official' else 'train'))
    estimates = {arm: macro(rows, arm) for arm in ('current', *controls)}
    if args.stage == 'official' and abs(estimates['current'] - .40674178096988606) > 1e-12:
        raise ValueError('Unexpected official baseline')
    comparisons = {arm: {reference: effect(rows, arm, reference)
                         for reference in ('current', 'frozen') if reference != arm and reference in estimates}
                   for arm in controls}
    if args.stage in ('screen', 'adapter-screen'):
        bounded = [arm for arm in TRAINED if arm in controls and all(
            min(delta['point'], delta['minimum_leave_one_domain_out']) > 0
            for delta in comparisons[arm].values())]
        selected = max(bounded, key=estimates.get, default=None)
    eligible = bool(selected and all(min(delta['point'], delta['minimum_leave_one_domain_out']) > 0
                                     for delta in comparisons[selected].values()))
    if args.stage == 'screen':
        comparisons['fusion']['independent'] = effect(rows, 'fusion', 'independent')
    return {'queries': expected, 'selected': selected, 'eligible': eligible, 'scores': estimates,
            'models': models, 'contrasts': comparisons, 'rank_fingerprint': fingerprint, 'per_query': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    screen = commands.add_parser('screen')
    for arg in ('frozen', *TRAINED):
        screen.add_argument(f'--{arg}', type=Path, required=True)
    adapter = commands.add_parser('adapter-screen')
    for arg in ('frozen', 'fusion'):
        adapter.add_argument(f'--{arg}', type=Path, required=True)
    confirm = commands.add_parser('confirm')
    confirm.add_argument('--frozen', type=Path, required=True)
    official = commands.add_parser('official')
    for command in (confirm, official):
        for arg in ('selection', 'scores'):
            command.add_argument(f'--{arg}', type=Path, required=True)
    for command in (screen, adapter, confirm, official):
        for arg in ('data-root', 'output'):
            command.add_argument(f'--{arg}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path.')
    result = evaluate(args)
    write_gz(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_query'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
