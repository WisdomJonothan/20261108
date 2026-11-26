"""Retain the first twenty candidates and allocate ten slots by scorer disagreement."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import POLICIES, blend, candidate_judges
from eval_temporal_condition_intervention import evaluate
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from replay_passage_admission import observations
from retrieve_union_with_pointwise_scan import REVISION


ARMS = ('current', 'select_tail_slots')


def replay(args):
    source = index({'per_query': load(args.scopes)['queries']})
    prior, reference = (index(load(path)) for path in (args.consensus, args.reference))
    pointwise = candidate_judges(args.pointwise)[0]['pretrained']
    predictions, models, _ = candidate_judges(args.scan)
    if models['pretrained']['revision'] != REVISION:
        raise ValueError('The candidate allocation requires the pinned 0.6B scan')
    scan = predictions['pretrained']
    evidence = observations(args.evidence)
    if any(part.keys() != source.keys() for part in (prior, reference, pointwise, scan, evidence)):
        raise ValueError('The comparisons must use identical queries')
    result = []
    for key, row in source.items():
        original = row['original']
        scores = pointwise[key]['scores']
        small = scan[key]['scores']
        if len(original) != 60 or len(set(original)) != 60 or not set(original) <= scores.keys() & small.keys():
            raise ValueError(f'All original candidates must have both scores for {key}')
        policy = POLICIES['adaptive30_free']
        votes = prior[key]['rankings']
        first = votes['pointwise']
        shared_votes = [votes[name] for name in ('current', 'reasoned', 'verified')]
        frozen = reference[key]['rankings']['verified_0.5']
        if blend(original[:30], {}, scores, policy) != first:
            raise ValueError(f'The replay must reproduce the original pointwise vote for {key}')
        if head(reciprocal(shared_votes + [first], 60), first, evidence[key], True, .5) != frozen:
            raise ValueError(f'The replay must reproduce the final ranking for {key}')
        assigned = set(sorted(original[20:], key=small.__getitem__, reverse=True)[:10])
        allocation = original[:20] + [doc for doc in original[20:] if doc in assigned]
        winner = blend(allocation, {}, scores, policy)
        updated = head(reciprocal(shared_votes + [winner], 60), first, evidence[key], True, .5)
        result.append(
            {field: row[field] for field in ('qid', 'domain', 'cohort', 'split', 'actual_reference')}
            | {'rankings': dict(zip(ARMS, (frozen, updated), strict=True)),
               'allocation': allocation,
               'beyond_original_first_30': len(assigned.intersection(original[30:]))}
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scopes', 'consensus', 'reference', 'data-root', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('scan', 'pointwise', 'evidence'):
        parser.add_argument(f'--{name}', type=Path, nargs='+', required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--reserve-ids', type=Path)
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or (args.confirmation and not args.selection):
        parser.error('Use a fresh output and a frozen rule for confirmation')
    output = evaluate(args, replay, ARMS)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(output, handle, separators=(',', ':'))


if __name__ == '__main__':
    main()
