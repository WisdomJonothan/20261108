"""Judge three new challengers against a fixed incumbent after prefix abstention."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_ce_disagreement_actions import prepare_text, run_judge
from replay_anchor_fallback import index
from replay_position_discount import bootstrap, eligible


ARMS = ('external6', 'rank_union', 'supervised')
ANCHOR_SLOT = 5


def build(root, data_root, folds):
    artifacts = root / 'results/replacement_v2'
    analysis = root / 'results/analysis'
    dev = 'dev' in folds
    report_split = 'dev' if dev else 'oof'
    label_split = 'dev' if dev else 'train'
    reference = index(load(analysis / f'position_oracle_{report_split}_20261124.json'))
    shortlists = index({'per_query': load(analysis / 'challenger_shortlist_20261124.json')['per_query'][report_split]})
    labels = {}
    rows = []
    for fold in folds:
        suffix = 'dev' if dev else f'train_{fold}'
        cached_rows = index({'per_query': load(artifacts / f'expanded3_fixed_{suffix}.json')['budgets']['18']['per_query']})
        manifest = load(artifacts / f'candidates_{suffix}.json')
        for item in manifest['queries']:
            domain, qid = key = item['domain'], item['qid']
            shortlist = shortlists[key]
            if shortlist['fold'] != fold:
                raise ValueError(f'Shortlist for {key} was not scored on {fold}')
            if domain not in labels:
                labels[domain] = read_qrels(data_root / domain / f'qrels_{label_split}.txt')
            gold = labels[domain][qid]
            top = item['frozen_top10']
            base = ndcg10(top, gold)
            if abs(base - reference[key]['baseline']) > 1e-10 or abs(base - shortlist['baseline']) > 1e-10:
                raise ValueError(f'Inconsistent original ranking for {key}')
            external = reference[key]['actions']['external']
            prefix = external[:3]
            routed = not eligible(prefix)
            arms = dict.fromkeys(ARMS, external)
            new_actions = {}
            if routed:
                seen = {action['challenger'] for action in prefix} | set(top)
                known = {(action['slot'], action['challenger']): action for action in cached_rows[key]['actions']}
                for name in ARMS[1:]:
                    selected = [doc for doc in shortlist[name]['six'] if doc not in seen][:3]
                    if len(selected) != 3 or len(set(selected)) != 3:
                        raise ValueError(f'Expected three distinct unseen challengers for {key}')
                    suffix_actions = []
                    for doc in selected:
                        pair_key = ANCHOR_SLOT, doc
                        if pair_key in known:
                            action = known[pair_key]
                        else:
                            candidate_rank = top.copy()
                            incumbent = candidate_rank[ANCHOR_SLOT - 1]
                            candidate_rank[ANCHOR_SLOT - 1] = doc
                            action = {'slot': ANCHOR_SLOT, 'incumbent': incumbent, 'challenger': doc,
                                      'delta': ndcg10(candidate_rank, gold) - base}
                            action = new_actions.setdefault(doc, action)
                        suffix_actions.append(action)
                    arms[name] = prefix + suffix_actions
            rows.append({
                'domain': domain, 'qid': qid, 'fold': fold, 'query': item['query'],
                'baseline': base, 'reachable_zero': shortlist['reachable_zero'],
                'routed': routed, 'arms': arms, 'treatment': list(new_actions.values()),
                'external_delta': reference[key]['policies']['external6']['delta'],
            })
    if len({(r['domain'], r['qid']) for r in rows}) != len(rows):
        raise ValueError('Evaluation contains duplicate queries')
    return rows


def outcome(actions, scored):
    result = {
        'oracle_delta': max(0., *(a['delta'] for a in actions)),
        'unique_challengers': len({a['challenger'] for a in actions}),
    }
    if scored:
        accepted = eligible(actions)
        chosen = max(accepted, key=lambda a: a['mp'], default=None)
        result |= {'delta': chosen['delta'] if chosen else 0., 'choice': chosen,
                   'accepted_oracle_delta': max([0.] + [a['delta'] for a in accepted])}
    return result


def summarize(rows, scored):
    baseline = bootstrap([(r['domain'], r['baseline']) for r in rows])['point']
    output = {'queries': len(rows), 'routed': sum(r['routed'] for r in rows), 'fresh_pairs': sum(r['fresh_pairs'] for r in rows),
              'dense_reachable_zeroes': sum(r['reachable_zero'] for r in rows),
              'routed_dense_reachable_zeroes': sum(r['reachable_zero'] and r['routed'] for r in rows), 'arms': {}}
    for name in ARMS:
        details = [r['outcomes'][name] for r in rows]
        reachable = [r for r in rows if r['reachable_zero'] and r['outcomes'][name]['oracle_delta'] > 1e-12]
        bound = bootstrap([(r['domain'], d['oracle_delta']) for r, d in zip(rows, details)])
        arm = {'oracle': baseline + bound['point'], 'zeroes_with_beneficial_comparison': len(reachable),
               'mean_unique_challengers': sum(d['unique_challengers'] for d in details) / len(details)}
        if scored or name == 'external6':
            gain = bootstrap([(r['domain'], d['delta']) for r, d in zip(rows, details)])
            arms_change = bootstrap([(r['domain'], d['delta'] - r['external_delta']) for r, d in zip(rows, details)])
            counts = Counter(
                'abstained' if d['choice'] is None else
                'helpful' if d['delta'] > 1e-12 else 'harmful' if d['delta'] < -1e-12 else 'neutral'
                for d in details
            )
            arm |= {
                'ndcg': baseline + gain['point'], 'versus_external6': arms_change, 'decisions': dict(counts),
                'zeroes_with_beneficial_accepted_comparison': sum(
                    r['outcomes'][name]['accepted_oracle_delta'] > 1e-12 for r in reachable),
                'zeroes_recovered': sum(r['outcomes'][name]['delta'] > 1e-12 for r in reachable),
                'new_zeroes': sum(r['baseline'] > 1e-12 and r['baseline'] + d['delta'] <= 1e-12
                                  for r, d in zip(rows, details)),
            }
        output['arms'][name] = arm
    if scored:
        output['supervised_vs_rank_union'] = bootstrap([
            (r['domain'], r['outcomes']['supervised']['delta'] - r['outcomes']['rank_union']['delta']) for r in rows
        ])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--folds', nargs='+', choices=('f0', 'f1', 'f2', 'dev'), required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Specify a new output path')
    if 'dev' in args.folds and args.folds != ['dev']:
        parser.error('Evaluate official dev separately from the training folds')
    data_root = args.data_root or args.root / 'data/track1_tempo'
    rows = build(args.root, data_root, args.folds)
    stats = None
    if args.model:
        pending = defaultdict(list)
        for row in rows:
            if row['treatment']:
                pending[row['domain']].append(row)
        for domain, selected in pending.items():
            prepare_text(selected, data_root / domain)
        stats = run_judge([row for selected in pending.values() for row in selected], args.model)
    records = []
    for row in rows:
        outcomes = {name: outcome(row['arms'][name], bool(stats is not None or name == 'external6')) for name in ARMS}
        if abs(outcomes['external6']['delta'] - row['external_delta']) > 1e-12:
            raise ValueError(f"Frozen reference disagrees for {row['qid']}")
        record = {k: v for k, v in row.items() if k not in ('query', 'pairs', 'arms', 'treatment')}
        record |= {
            'outcomes': outcomes, 'fresh_pairs': len(row['treatment']),
            'suffix_challengers': {name: [a['challenger'] for a in row['arms'][name][3:]] for name in ARMS},
        }
        if row['treatment'] and stats is not None:
            record['fresh_judgments'] = row['treatment']
        records.append(record)
    result = {
        'protocol': 'Keep the first three original comparisons. If they all fail the acceptance gate, '
                    'judge three unseen candidates against the original document at rank five. '
                    'Select them from the existing held-out rank-union or supervised shortlist on the '
                    'same dense top 100. Keep the original suffix if any prefix comparison passes. '
                    'All arms use six comparisons, both orders, and at most one replacement. '
                    'The shared inference run reuses unchanged, identical pairs across arms. '
                    'Qrels are used only for scoring and diagnostics.',
        'folds': args.folds, 'inference': stats, 'summary': summarize(records, stats is not None),
        'per_fold': {fold: summarize([r for r in records if r['fold'] == fold], stats is not None)
                     for fold in args.folds},
        'per_query': records,
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
