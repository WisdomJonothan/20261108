"""Reconstruct the frozen ranking stages on disjoint training questions."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import POLICIES, blend, candidate_judges
from eval_temporal_evidence_categories import ranking
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_document_repair import fit, label, predict, prepare
from replay_evidence_ranking_consensus import reciprocal


def repair(args):
    training = prepare(args.training_manifest, args.training_scores)
    confirmation = prepare(args.manifest, args.pointwise)
    fit_keys = {(r['domain'], r['qid']) for r in training}
    confirm_keys = {(r['domain'], r['qid']) for r in confirmation}
    if len(fit_keys) != 186 or len(confirm_keys) != args.query_count or fit_keys & confirm_keys:
        raise ValueError('Fit and confirmation require fixed disjoint query samples')
    estimates = predict(confirmation, fit(label(training, 'train', args.data_root)))
    return {'choices': dict.fromkeys({r['cohort'] for r in estimates}, 'restore_0'),
            'oof_per_query': estimates}


def consensus(args):
    wide = index({'per_query': load(args.wide)['queries']})
    grades = index({'per_query': [r for f in args.reasoner for r in load(f)['per_query']]})
    earlier = index({'per_query': load(args.repair)['oof_per_query']})
    pointwise = candidate_judges(args.pointwise)[0]['pretrained']
    if not wide.keys() == grades.keys() == earlier.keys() == pointwise.keys() or len(wide) != args.query_count:
        raise ValueError('The same confirmation questions are required at each pipeline stage')
    rows, edits = [], []
    for key, query in wide.items():
        documents = query['docs']
        scores = pointwise[key]['scores']
        obs = grades[key]['observations']
        current = earlier[key]['rankings']['restore_0']
        shortlist = list(obs)
        if len(shortlist) != 14 or shortlist[:10] != current or set(documents) != scores.keys():
            raise ValueError(f'Invalid original evidence pool for {key}')
        point = blend(documents, scores, scores, POLICIES['adaptive30_free'])
        reasoned = ranking(shortlist, [obs[d]['reasoned'] for d in shortlist])
        verified = ranking(shortlist, [obs[d]['reasoned'] if obs[d]['verified'] else obs[d]['direct']
                                      for d in shortlist])
        combined = reciprocal((current, reasoned, verified, point), 60)
        latest = head(combined, point, obs, True, .5)
        metadata = {k: query[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')}
        rows.append(metadata | {'rankings': dict(current=current, reasoned=reasoned, verified=verified,
                                                 pointwise=point, agreement_offset60=combined)})
        edits.append(metadata | {'rankings': {'current': current, 'verified_0.5': latest}})
    return {'per_query': rows}, {'per_query': edits}


def save(path, output):
    with path.open('x', encoding='utf-8') as file:
        json.dump(output, file, separators=(',', ':'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    for name in ('repair', 'consensus'):
        command = stages.add_parser(name)
        command.add_argument('--pointwise', type=Path, nargs='+', required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--query-count', type=int, default=155)
        options = ('manifest', 'training-manifest', 'data-root') if name == 'repair' else ('wide', 'repair', 'head-output')
        for option in options:
            command.add_argument(f'--{option}', type=Path, required=True)
        command.add_argument('--training-scores' if name == 'repair' else '--reasoner',
                             type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if args.query_count < 1:
        parser.error('Expected a positive confirmation sample size')
    if args.stage == 'repair':
        save(args.output, repair(args))
    else:
        ranks, edited = consensus(args)
        save(args.output, ranks)
        save(args.head_output, edited)


if __name__ == '__main__':
    main()
