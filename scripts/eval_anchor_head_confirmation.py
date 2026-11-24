"""Confirm one anchor-proposed head promotion with a fresh two-order comparison."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_ce_disagreement_actions import prepare_text, run_judge
from replay_anchor_fallback import index, protect_first_three
from replay_position_discount import bootstrap, eligible


def propose(protected, anchor_top):
    for slot in range(3):
        incumbent, challenger = protected[slot], anchor_top[slot]
        if incumbent == challenger:
            continue
        proposed = protected.copy()
        if challenger in proposed:
            lower_slot = proposed.index(challenger)
            proposed[lower_slot] = incumbent
        proposed[slot] = challenger
        action = {'slot': slot + 1, 'incumbent': incumbent, 'challenger': challenger,
                  'type': 'exchange' if challenger in protected else 'replacement'}
        return action, proposed
    return None, protected


def build(root, data_root, source, capped, reference, split):
    if source.keys() != capped.keys() or not source.keys() <= reference.keys():
        raise ValueError('Source, cap, and reference files disagree on matched queries')
    suffixes = ('dev',) if split == 'dev' else ('train_f0', 'train_f1', 'train_f2')
    directory = root / 'results/replacement_v2'
    questions = {
        (row['domain'], row['qid']): row
        for suffix in suffixes for row in load(directory / f'candidates_{suffix}.json')['queries']
    }
    gold_by_domain = {
        domain: read_qrels(data_root / domain / f'qrels_{split}.txt')
        for domain in {domain for domain, _ in source}
    }
    rows = []
    for key, row in source.items():
        domain, qid = key
        original = row['arms']['base']['ranking']
        question = questions[key]
        if original != question['frozen_top10']:
            raise ValueError(f'Candidate manifest disagrees with the base for {key}')
        base_score = row['arms']['base']['ndcg']
        if abs(base_score - reference[key]['baseline']) > 1e-10:
            raise ValueError(f'Reference disagrees with the base for {key}')
        anchor_top = capped[key]['ranking']
        gold = gold_by_domain[domain][qid]
        if abs(ndcg10(anchor_top, gold) - capped[key]['cap20']) > 1e-10:
            raise ValueError(f'Cached ranking disagrees with qrels for {key}')
        external = reference[key]['policies']['external6']
        control = base_score + external['delta']
        protected, proposed, unprotected = control, control, control
        action = None
        if external['chosen'] is None:
            protected_top = protect_first_three(original, anchor_top)
            action, proposed_top = propose(protected_top, anchor_top)
            protected = ndcg10(protected_top, gold)
            proposed = ndcg10(proposed_top, gold)
            unprotected = capped[key]['cap20']
        rows.append({
            'domain': domain, 'qid': qid, 'query': question['query'],
            'external6': control, 'abstained': external['chosen'] is None,
            'protected': protected, 'unprotected': unprotected, 'always_take_proposal': proposed,
            'treatment': [action] if action else [],
        })
    return rows


def summarize(rows, stats):
    keys = ('external6', 'protected', 'unprotected', 'always_take_proposal')
    if stats is not None:
        keys += ('confirm_in_both_orders',)
    def comparison(arm, reference):
        return bootstrap([(r['domain'], r[arm] - r[reference]) for r in rows])

    proposed = [r for r in rows if r['treatment']]
    arms = {}
    for key in keys:
        effects = Counter(
            'helpful' if r[key] > r['protected'] + 1e-12 else
            'harmful' if r[key] < r['protected'] - 1e-12 else 'unchanged'
            for r in proposed
        )
        arms[key] = {
            'ndcg': bootstrap([(r['domain'], r[key]) for r in rows])['point'],
            'versus_protected': comparison(key, 'protected'),
            'proposed_query_effects_vs_protected': dict(effects),
        }
    result = {
        'queries': len(rows), 'external_abstentions': sum(r['abstained'] for r in rows),
        'proposals': len(proposed),
        'proposal_types': dict(Counter(r['treatment'][0]['type'] for r in proposed)),
        'proposal_slots': dict(Counter(r['treatment'][0]['slot'] for r in proposed)),
        'arms': arms,
        'oracle_delta_vs_protected': bootstrap(
            [(r['domain'], max(0., r['always_take_proposal'] - r['protected'])) for r in rows]),
    }
    if stats is not None:
        result['confirmations'] = sum(bool(eligible(r['treatment'])) for r in proposed)
        result['confirmation_vs_always_take'] = comparison('confirm_in_both_orders', 'always_take_proposal')
        result['additional_mean_prompts'] = stats.get('prompts', 0) / len(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, help='Defaults to data/track1_tempo under --root.')
    parser.add_argument('--model', type=Path, help='Omit to audit the proposals without scoring them.')
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--cap', type=Path, nargs='+', help='Cached cap20 rankings for full-pool inputs.')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.cap and len(args.input) != len(args.cap)) or args.output.exists():
        parser.error('Provide a new output path and, if supplied, one capped ranking per input.')
    data_root = args.data_root or args.root / 'data/track1_tempo'
    reference = index(load(args.reference))
    all_rows, seen, samples, splits = [], set(), {}, set()
    for i, path in enumerate(args.input):
        report = load(path)
        split = report.get('split', 'train')
        source = index(report)
        if seen.intersection(source):
            raise ValueError('Evaluation samples must be disjoint')
        seen.update(source)
        splits.add(split)
        if args.cap:
            capped = index(load(args.cap[i]))
        else:
            if report['pool_depth'] != 20:
                parser.error('Full-pool scores require a cap20 replay from --cap.')
            anchor_arms = {key: row['arms']['base_plus_anchor30'] for key, row in source.items()}
            capped = {key: {'ranking': arm['ranking'], 'cap20': arm['ndcg']} for key, arm in anchor_arms.items()}
        samples[path.name] = build(args.root, data_root, source, capped, reference, split)
        all_rows.extend(samples[path.name])
    if len(splits) != 1:
        raise ValueError('Training folds and official dev need separate reports')
    stats = None
    if args.model:
        by_domain = defaultdict(list)
        for row in all_rows:
            if row['treatment']:
                by_domain[row['domain']].append(row)
        for domain, rows in by_domain.items():
            prepare_text(rows, data_root / domain)
        stats = run_judge([row for rows in by_domain.values() for row in rows], args.model)
        for row in all_rows:
            row['confirm_in_both_orders'] = (row['always_take_proposal']
                                            if eligible(row['treatment']) else row['protected'])
    result = {
        'protocol': 'Keep accepted external6 decisions. On other queries, start with the first '
                    'three original documents and the anchor-fused tail. Propose only the '
                    'first differing head position from the unprotected anchor list. Exchange '
                    'the two positions if both documents are already present; otherwise replace '
                    'the incumbent. The direct judge confirms only if the challenger wins in '
                    'both orders and its mean probability exceeds 0.5. Qrels score outcomes.',
        'split': splits.pop(),
        'inference': stats,
        'summary': summarize(all_rows, stats),
        'samples': {
            name: summarize(rows, None if stats is None else {'prompts': 2 * sum(bool(r['treatment']) for r in rows)})
            for name, rows in samples.items()
        },
        'per_query': [{k: v for k, v in r.items() if k not in ('query', 'pairs')} for r in all_rows],
    }
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(result['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
