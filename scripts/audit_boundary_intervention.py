"""Audit boundary reweighting with train-group counts and fixed score snapshots."""
import argparse
import json
from pathlib import Path

import numpy as np

from lib_reteco import read_trec_run


def snapshot(docs, labels, scores, dense_rank):
    order = np.argsort(-np.asarray(scores), kind='stable')
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    positive = np.asarray(labels) > 0
    pairs = positive[:, None] & ~positive[None, :]
    active = pairs & ((rank[:, None] < 10) | (rank[None, :] < 10))
    targeted = pairs & (dense_rank[:, None] >= 10) & (dense_rank[None, :] < 10)
    discount = np.where(rank < 10, 1 / np.log2(rank + 2), 0.)
    weights = np.abs(discount[:, None] - discount[None, :]) * active
    total = weights.sum()
    share = float(weights[targeted].sum() / total) if total else 0.
    normalized_change = float(np.abs(
        weights * (1 + targeted) / max((weights * (1 + targeted)).sum(), 1e-12)
        - weights / max(total, 1e-12)).sum())
    return {'target_pairs': int(targeted.sum()), 'active_pairs': int(active.sum()),
            'active_target_pairs': int((active & targeted).sum()),
            'target_original_weight_share': share,
            'normalized_weight_l1_change': normalized_change,
            'no_change': bool(normalized_change < 1e-12)}


def pair_counts(ids, positive, dense_rank):
    rank = {doc: i for i, doc in enumerate(ids)}
    wrong = corrected = right = destroyed = 0
    for p in positive:
        for n in dense_rank:
            if n in positive or ((dense_rank[p] < 10) == (dense_rank[n] < 10)):
                continue
            if dense_rank[p] > dense_rank[n]:
                wrong += 1
                corrected += rank[p] < rank[n]
            else:
                right += 1
                destroyed += rank[p] > rank[n]
    return {'correction': corrected / wrong if wrong else None,
            'destruction': destroyed / right if right else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    split = json.loads((root / 'results/fusion100_split.json').read_text())
    heldout = {(d, q) for d, qs in split['eval'].items() for q in qs}
    baseline = 'fusion100_full_ce_s17_20261115'
    boundary = 'fusion100_full_boundary_s17_20261118'
    candidates = json.loads((root / 'results' / f'{baseline}_trainval_candidates.json').read_text())
    snapshots = []
    for row in candidates['rows']:
        dense_order = np.argsort(-np.asarray(row['dense']), kind='stable')
        dense_rank = np.empty(100, dtype=int)
        dense_rank[dense_order] = np.arange(100)
        labels = [row['gold'].get(doc, 0) for doc in row['docs']]
        snapshots.append({'domain': row['domain'], 'qid': row['qid'],
                          **snapshot(row['docs'], labels, row['ce'], dense_rank)})
    potential = []
    changes = {'ce': {}, 'fused': {}}
    details = []
    for domain in split['eval']:
        dense_train = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_train.trec')
        for line in (root / 'results' / f'fusion100_groups_{domain}.jsonl').read_text().splitlines():
            group = json.loads(line)
            if (domain, group['qid']) in heldout:
                continue
            ranks = {doc: i for i, (_, _, doc) in enumerate(dense_train[group['qid']][:100])}
            positives = [doc['id'] for doc in group['docs'] if doc['label'] > 0]
            negatives = [doc['id'] for doc in group['docs'] if doc['label'] == 0]
            count = sum(ranks[p] >= 10 and ranks[n] < 10 for p in positives for n in negatives)
            potential.append(count)
        gold = {}
        for line in (root / 'data/track1_tempo' / domain / 'qrels_dev.txt').read_text().splitlines():
            q, _, doc, relevance = line.split()
            if int(relevance) > 0:
                gold.setdefault(q, set()).add(doc)
        dense = read_trec_run(root / 'runs' / f'dense_mxbai_{domain}_dev.trec')
        for mode in changes:
            old = read_trec_run(root / 'runs' / f'{baseline}_{domain}_dev_{mode}.trec')
            new = read_trec_run(root / 'runs' / f'{boundary}_{domain}_dev_{mode}.trec')
            changes[mode][domain] = {'correction': [], 'destruction': [], 'relevant_top10_delta': []}
            for q in old:
                ranks = {doc: i for i, (_, _, doc) in enumerate(dense[q][:100])}
                old_ids = [doc for _, _, doc in old[q]]
                new_ids = [doc for _, _, doc in new[q]]
                assert set(old_ids) == set(new_ids) == set(ranks)
                pos = gold[q] & set(ranks)
                before = pair_counts(old_ids, pos, ranks)
                after = pair_counts(new_ids, pos, ranks)
                for metric in before:
                    if before[metric] is not None:
                        changes[mode][domain][metric].append(after[metric] - before[metric])
                relevant_delta = len(set(new_ids[:10]) & pos) - len(set(old_ids[:10]) & pos)
                changes[mode][domain]['relevant_top10_delta'].append(relevant_delta)
                details.append({'domain': domain, 'qid': q, 'mode': mode,
                                'before': before, 'after': after, 'relevant_top10_delta': relevant_delta})
    summary = {}
    for mode, domains in changes.items():
        summary[mode] = {}
        for metric in ['correction', 'destruction', 'relevant_top10_delta']:
            means = [float(np.mean(v[metric])) for v in domains.values() if v[metric]]
            summary[mode][metric] = {'domain_macro_delta': float(np.mean(means)),
                                    'per_domain': {d: float(np.mean(v[metric])) for d, v in domains.items() if v[metric]}}
    result = {
        'scope': 'Static target coverage for 857 train groups; active-pair and normalization audit on baseline best checkpoint heldout214 snapshots. Actual sampled-window training trajectory was not logged. Dev pair analysis uses qrels positives versus other pool documents as zero.',
        'training_groups': len(potential), 'training_groups_with_target_pairs': sum(n > 0 for n in potential),
        'snapshot_queries': len(snapshots),
        'snapshot_no_normalized_change': sum(r['no_change'] for r in snapshots),
        'snapshot_no_active_target': sum(r['active_target_pairs'] == 0 for r in snapshots),
        'snapshot_mean_normalized_weight_l1_change': float(np.mean([r['normalized_weight_l1_change'] for r in snapshots])),
        'dev_boundary_pair_changes': summary, 'snapshot_details': snapshots, 'dev_details': details}
    assert len(potential) == 857 and len(snapshots) == 214 and len(details) == 1038
    output = root / 'results/boundary_intervention_audit.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['snapshot_details', 'dev_details']}, indent=2))


if __name__ == '__main__':
    main()
