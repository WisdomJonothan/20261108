"""Isolate document admission from ranking on cached shared-reference scores."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from lib_reteco import read_trec_run
from replay_anchor_cap import z
from replay_anchor_fallback import discounted_changes, index
from replay_position_discount import bootstrap


ARMS = ('start', 'reorder_same_ten', 'one_fixed', 'one_reordered', 'unrestricted')


def original_scores(root, split, sample):
    group = defaultdict(list)
    for key, row in sample.items():
        group[key[0]].append(row)
    scores = {}
    for domain, rows in sorted(group.items()):
        dense = read_trec_run(root / f'dense_mxbai_{domain}_{split}.trec')
        for fold in sorted({r['fold'] for r in rows}):
            name = f'trainfuse_{fold}_{domain}_train' if split == 'train' else f'sub_sft_e2bc_{domain}_dev'
            fused = read_trec_run(root / f'{name}.trec')
            for row in (r for r in rows if r['fold'] == fold):
                qid = row['qid']
                membership = row['scores'].keys()
                base = {doc: score for _, score, doc in fused[qid] if doc in membership}
                top = [doc for _, _, doc in fused[qid][:10]]
                if top != row['arms']['base']['ranking'] or base.keys() != membership:
                    raise ValueError(f'Original fused run disagrees with the anchor cache for {domain}/{qid}')
                scores[domain, qid] = [[doc, base[doc]] for _, _, doc in dense[qid][:100] if doc in membership]
    return scores


def rankings(start, ordered):
    incumbent_set = set(start)
    outside = next(doc for doc in ordered if doc not in incumbent_set)
    weakest = next(doc for doc in reversed(ordered) if doc in incumbent_set)
    fixed = start.copy()
    if ordered.index(outside) < ordered.index(weakest):
        fixed[fixed.index(weakest)] = outside
    one_set = set(fixed)
    return {
        'start': start,
        'reorder_same_ten': [doc for doc in ordered if doc in incumbent_set],
        'one_fixed': fixed,
        'one_reordered': [doc for doc in ordered if doc in one_set],
        'unrestricted': ordered[:10],
    }


def paired_delta(differences):
    by_domain = defaultdict(list)
    for domain, delta in differences:
        by_domain[domain].append(delta)
    per_domain = {domain: mean(values) for domain, values in sorted(by_domain.items())}
    means = np.array(list(per_domain.values()))
    rng = np.random.default_rng(20261125)
    resampled = means[rng.integers(len(means), size=(10000, len(means)))].mean(axis=1)
    return bootstrap(differences) | {
        'domain_bootstrap_ci95': np.quantile(resampled, [.025, .975]).tolist(),
        'per_domain': per_domain,
        'improved_queries': sum(d > 1e-12 for _, d in differences),
        'declined_queries': sum(d < -1e-12 for _, d in differences),
    }



def summarize(rows, start_name):
    active = sum(r['active'] for r in rows)
    scores = {arm: bootstrap([(r['domain'], r['scores'][arm]) for r in rows])['point'] for arm in ARMS}
    scores['external6'] = bootstrap([(r['domain'], r['external6']) for r in rows])['point']
    pairs = (
        ('one_fixed', 'start'), ('reorder_same_ten', 'start'), ('one_reordered', 'one_fixed'),
        ('unrestricted', 'one_reordered'), ('unrestricted', 'start'),
    )
    comparisons = {}
    for after, before in pairs:
        differences = [(r['domain'], r['scores'][after] - r['scores'][before]) for r in rows]
        comparisons[f'{after}_vs_{before}'] = paired_delta(differences)

    def macro(values):
        by_domain = defaultdict(list)
        for row, value in zip(rows, values):
            by_domain[row['domain']].append(value)
        return mean(mean(group) for group in by_domain.values())

    effects = {
        arm: {
            name: macro([row['effects'][arm][name] for row in rows])
            for name in ('added_documents', 'removed_documents', 'retained_documents')
        }
        for arm in ARMS if arm != 'start'
    }
    for arm, effect in effects.items():
        if abs(sum(effect.values()) - (scores[arm] - scores['start'])) > 1e-10:
            raise ValueError(f'nDCG accounting does not sum for {arm}')
    zeroes = [r for r in rows if r['scores']['start'] == 0.0]
    return {
        'start': start_name, 'active_queries': active, 'queries': len(rows), 'domains': len({r['domain'] for r in rows}),
        'ndcg': scores, 'paired_deltas': comparisons, 'discounted_effects_vs_start': effects,
        'proposal_oracle': macro([max(row['scores'].values()) for row in rows]),
        'start_zeroes': len(zeroes),
        'start_zeroes_with_positive_in_pool': sum(r['has_pool_positive'] for r in zeroes),
        'recovered_start_zeroes': {arm: sum(r['scores'][arm] > 0.0 for r in zeroes) for arm in ARMS},
        'single_swap_queries': sum(r['replacements']['one_fixed'] == 1 for r in rows),
        'unrestricted_more_than_one': sum(r['replacements']['unrestricted'] > 1 for r in rows),
    }


def ordered_scores(base, anchor):
    docs = [doc for doc, _ in base]
    frozen_z = z([score for _, score in base])
    anchor_z = z([anchor[doc] for doc in docs])
    return [docs[i] for i in np.argsort(-(.7 * frozen_z + .3 * anchor_z), kind='stable')]



def replay(sample, reference, pool_scores, labels, start_name, source_depth, depth):
    results = []
    for key, row in sample.items():
        domain, qid = key
        gold = labels[domain][qid]
        original = row['arms']['base']['ranking']
        cached = reference[key]
        external6 = original.copy()
        policy = cached['policies']['external6']
        if action := policy['chosen']:
            slot = action['slot'] - 1
            if external6[slot] != action['incumbent'] or action['challenger'] in external6:
                raise ValueError(f'Competition action disagrees with the original list for {key}')
            external6[slot] = action['challenger']
        frozen_ndcg, competition_ndcg = ndcg10(original, gold), ndcg10(external6, gold)
        if abs(frozen_ndcg - row['arms']['base']['ndcg']) > 1e-10 or abs(
                competition_ndcg - cached['baseline'] - policy['delta']) > 1e-10:
            raise ValueError(f'Cached scores disagree with qrels for {key}')
        base = pool_scores[key]
        docs = [doc for doc, _ in base]
        if len(docs) != source_depth or len(set(docs)) != source_depth or set(docs) != row['scores'].keys():
            raise ValueError(f'Candidate pool and model scores disagree for {key}')
        ordered = ordered_scores(base, row['scores'])
        if ordered[:10] != row['arms']['base_plus_anchor30']['ranking']:
            raise ValueError(f'Reconstructed ranking disagrees with cached policy for {key}')
        if depth != source_depth:
            members = set(list(dict.fromkeys(original + docs))[:depth])
            docs = [doc for doc in docs if doc in members]
            ordered = ordered_scores([(doc, score) for doc, score in base if doc in members], row['scores'])
        start = original if start_name == 'frozen' else external6
        active = start_name != 'abstention' or action is None
        if active and not set(start) <= set(docs):
            raise ValueError(f'The starting list includes a document outside the pool for {key}')
        arms = rankings(start, ordered) if active else dict.fromkeys(ARMS, start)
        if len(set(arms['one_fixed']) - set(start)) > 1 or set(arms['one_fixed']) != set(arms['one_reordered']):
            raise ValueError(f'One-swap arms must have the same document membership for {key}')
        results.append({
            'domain': domain, 'qid': qid, 'fold': row['fold'], 'external6': competition_ndcg, 'active': active,
            'has_pool_positive': bool(set(docs) & gold),
            'scores': {arm: ndcg10(order, gold) for arm, order in arms.items()},
            'replacements': {arm: len(set(order) - set(start)) for arm, order in arms.items()},
            'effects': {arm: {k: value for k, value in discounted_changes(start, order, gold).items()
                              if k != 'rank_delta'}
                        for arm, order in arms.items() if arm != 'start'},
            'rankings': arms,
        })
    return results


def compare_cache(rows, previous):
    baseline = index(previous)
    matched = []
    for row in rows:
        key = row['domain'], row['qid']
        other = baseline[key]
        if row['fold'] != other['fold'] or row['rankings']['start'] != other['rankings']['start']:
            raise ValueError(f'The comparison reports disagree on the starting list for {key}')
        matched.append((row['domain'], row['scores'], other['scores']))

    comparisons = {}
    for arm in ARMS:
        differences = [(domain, new[arm] - old[arm]) for domain, new, old in matched]
        comparisons[arm] = paired_delta(differences) | {
            'prior_score': bootstrap([(domain, old[arm]) for domain, _, old in matched])['point'],
        }

    interactions = {}
    for after, before in (('one_reordered', 'one_fixed'), ('unrestricted', 'one_reordered')):
        differences = [(domain, (new[after] - new[before]) - (old[after] - old[before]))
                       for domain, new, old in matched]
        interactions[f'{after}_vs_{before}'] = paired_delta(differences)
    return {'same_policy': comparisons, 'edit_scope_interaction': interactions}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--pool-scores', type=Path)
    parser.add_argument('--protected-depth', type=int,
                        help='Replay a smaller pool that preserves the frozen top ten.')
    parser.add_argument('--compare', type=Path, help='Prior edit-scope output for a paired model comparison.')
    parser.add_argument('--write-pool-scores', type=Path)
    parser.add_argument('--starts', choices=('frozen', 'competition', 'abstention'), nargs='+', default=['frozen'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if bool(args.runs_root) == bool(args.pool_scores):
        parser.error('Provide either --runs-root or --pool-scores.')
    reports = [load(path) for path in args.inputs]
    split, source_depth = reports[0].get('split', 'train'), reports[0].get('pool_depth', 100)
    depth = args.protected_depth if args.protected_depth is not None else source_depth
    if not 10 < depth <= source_depth:
        parser.error('Replay depth must be greater than ten and within the cached pool.')
    if any((r.get('split', 'train'), r.get('pool_depth', 100)) != (split, source_depth) for r in reports):
        parser.error('Each input needs the same split and pool depth.')
    sample = index({'per_query': [row for r in reports for row in r['per_query']]})
    reference = index(load(args.reference))
    if args.pool_scores:
        pool_scores = {key: row['frozen_scores'] for key, row in index(load(args.pool_scores)).items()}
    else:
        pool_scores = original_scores(args.runs_root, split, sample)
    if not (sample.keys() <= pool_scores.keys() and sample.keys() <= reference.keys()):
        raise ValueError('Pool scores and policy references do not cover the supplied queries')
    labels = {domain: read_qrels(args.data_root / domain / f'qrels_{split}.txt')
              for domain in sorted({domain for domain, _ in sample})}
    result = {
        'protocol': 'Use the same cached scores and 70:30 standardized blend in every edit arm. '
                    'Admit the best outsider if it outranks the weakest incumbent. The two one-swap '
                    'arms use identical membership. Reordering the same ten keeps the original membership. '
                    'The abstention run keeps accepted six-pair decisions. Tie-breaking follows the '
                    'dense rank. Qrels are used exclusively for scoring, including a diagnostic '
                    'oracle that picks the best output among the five already-scored lists.',
        'split': split, 'pool_depth': depth,
        'anchor_prompts_per_query': 2 * (depth - 1),
        'inputs': [path.name for path in args.inputs], 'starts': {},
    }
    if depth != source_depth:
        result['source_pool_depth'] = source_depth
    previous = load(args.compare) if args.compare else None
    if previous and (previous['split'], previous['pool_depth']) != (split, depth):
        raise ValueError('Paired model comparison requires the same split and pool depth')
    for name in args.starts:
        rows = replay(sample, reference, pool_scores, labels, name, source_depth, depth)
        result['starts'][name] = {'summary': summarize(rows, name), 'per_query': rows}
        if previous:
            result['starts'][name]['paired_model_comparison'] = compare_cache(rows, previous['starts'][name])
        if len(reports) > 1:
            by_key = index({'per_query': rows})
            groups = {}
            for path, report in zip(args.inputs, reports):
                group = [by_key[key] for key in index(report)]
                groups[path.name] = summarize(group, name)
                if previous:
                    groups[path.name]['paired_model_comparison'] = compare_cache(group, previous['starts'][name])
            result['starts'][name]['by_input'] = groups
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, separators=(',', ':'))
        stream.write('\n')
    if args.write_pool_scores:
        with args.write_pool_scores.open('x', encoding='utf-8') as stream:
            json.dump({'per_query': [{'domain': d, 'qid': q, 'frozen_scores': values}
                                     for (d, q), values in pool_scores.items()]}, stream, separators=(',', ':'))
            stream.write('\n')
    for item in result['starts'].values():
        print(json.dumps(item['summary'], indent=1))


if __name__ == '__main__':
    main()
