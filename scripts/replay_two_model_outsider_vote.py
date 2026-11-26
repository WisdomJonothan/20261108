"""Grant a wider-pool candidate a second vote only when a second model supports it."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import POLICIES, blend, candidate_judges
from eval_temporal_condition_intervention import evaluate
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_passage_admission import observations
from replay_wide_pointwise_vote import replay as replay_fixed, retained_candidate_list
from retrieve_union_with_pointwise_scan import REVISION


ARMS = ('current', 'two_model_outsider_vote')


def fuse(votes, secondary, eligible):
    weights = defaultdict(float)
    for ranking in votes:
        for position, doc in enumerate(ranking, start=1):
            weights[doc] += 1. / (60 + position)
    for position, doc in enumerate(secondary, start=1):
        if doc in eligible:
            weights[doc] += 1. / (60 + position)
    return sorted(weights, key=lambda doc: -weights[doc])[:10]


def replay(args):
    pools = index({'per_query': load(args.manifest)['queries']})
    proof = observations(args.evidence)
    consensus = index(load(args.consensus))
    scores = candidate_judges(args.pointwise)[0]['pretrained']
    proposals, metadata, _ = candidate_judges(args.scan)
    if metadata['pretrained']['revision'] != REVISION:
        raise ValueError('Use the pinned 0.6B candidate scan')
    small = proposals['pretrained']
    output = []
    for row in replay_fixed(args):
        key = row['domain'], row['qid']
        scope = pools[key]
        old = scope['original']
        docs = retained_candidate_list(old, scope['wide'])
        raw = scores[key]['scores']
        established = consensus[key]['rankings']
        others = [established[name] for name in ('current', 'reasoned', 'verified')]
        primary = blend(docs, {}, raw, POLICIES['adaptive30_free'])
        secondary = sorted(docs, key=small[key]['scores'].__getitem__, reverse=True)[:10]
        eligible = (set(primary) & set(secondary)) - set(old)
        revised = fuse([*others, primary], secondary, eligible)
        rankings = {name: row['rankings'][name] for name in ('current', 'local_candidate_scale')}
        rankings[ARMS[1]] = head(revised, established['pointwise'], proof[key], True, .5)
        if not eligible and rankings[ARMS[1]] != rankings['local_candidate_scale']:
            raise ValueError('Queries without eligible outsiders must keep the matched control')
        output.append({field: row[field] for field in ('domain','qid','cohort','split','actual_reference')}
                      | {'rankings': rankings, 'eligible': sorted(eligible)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','reference','consensus','data-root','output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name in ('pointwise','evidence','scan'):
        parser.add_argument(f'--{name}', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    args.retain_position_prior = True
    args.guard_admission = args.stable_across_scales = args.protect_membership = args.taper_vote = False
    args.selection = None
    args.confirmation = False
    output = evaluate(args, replay, ARMS)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(output, handle, separators=(',', ':'))


if __name__ == '__main__':
    main()
