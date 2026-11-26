"""Compare equal-size 8B candidate pools before four-list ranking consensus."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges
from eval_temporal_condition_intervention import evaluate
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from replay_passage_admission import observations
from retrieve_union_with_pointwise_scan import REVISION


ARMS = ('current', 'replace_pointwise_with_wider_scope')


def prepare(args):
    rows = load(args.manifest)['queries']
    reference = index(load(args.reference))
    predictions, identities, _ = candidate_judges(args.initial_scan)
    if identities['pretrained']['revision'] != REVISION:
        raise ValueError('Use the pinned broad-scan checkpoint')
    initial = predictions['pretrained']
    cached = candidate_judges(args.pointwise)[0]['pretrained']
    output, missing = [], []
    for row in rows:
        key = row['domain'], row['qid']
        original = row['original']
        prior = reference[key]['rankings']['verified_0.5']
        early = initial[key]['scores']
        seen = cached[key]['scores']
        if len(original) != len(set(original)) or len(original) != 60 or not set(prior) <= set(original):
            raise ValueError(f'The original sixty documents need the same frozen baseline for {key}')
        if early.keys() != set(row['docs']) or not set(original) <= seen.keys():
            raise ValueError(f'Missing matching stored scores for {key}')
        outside = [doc for doc in row['docs'] if doc not in prior]
        wide = prior + sorted(outside, key=early.__getitem__, reverse=True)[:50]
        missing_docs = [doc for doc in wide if doc not in seen]
        output.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                      {'original':original, 'wide':wide})
        if missing_docs:
            missing.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference','query')} |
                           {'docs':missing_docs})
    with args.missing_manifest.open('x', encoding='utf8') as file:
        json.dump({'queries':missing},file,separators=(',',':'))
    print(len(missing),'queries,',sum(len(row['docs']) for row in missing),'uncached 8B pairs',flush=True)
    return {'queries':output}


def replay(args):
    source = index({'per_query':load(args.manifest)['queries']})
    prior, reference = (index(load(path)) for path in (args.consensus,args.reference))
    point = candidate_judges(args.pointwise)[0]['pretrained']
    evidence = observations(args.evidence)
    if not source.keys()==prior.keys()==reference.keys()==point.keys()==evidence.keys():
        raise ValueError('Every prediction source must cover the same queries')
    result = []
    for key,row in source.items():
        old, wide = row['original'], row['wide']
        old_votes = prior[key]['rankings']
        basis = [old_votes[arm] for arm in ('current','reasoned','verified')]
        original_pointwise = old_votes['pointwise']
        frozen = reference[key]['rankings']['verified_0.5']
        judgments = evidence[key]
        scores = point[key]['scores']
        if len(wide) != len(set(wide)) or len(wide) != len(old) or not set(old+wide) <= scores.keys():
            raise ValueError(f'Unexpected equal-budget candidate identities for {key}')
        if head(reciprocal(basis+[original_pointwise],60), original_pointwise, judgments, True,.5) != frozen:
            raise ValueError(f'Unexpected frozen baseline for {key}')
        original = sorted(old, key=scores.__getitem__, reverse=True)[:10]
        expanded = sorted(wide, key=scores.__getitem__, reverse=True)[:10]
        rankings = {'current': frozen, 'standalone_original60':original, 'standalone_wider60':expanded}
        for arm,vote in (('replace_pointwise_with_original_scope',original),(ARMS[1],expanded)):
            rankings[arm] = head(reciprocal(basis+[vote],60),original_pointwise,judgments,True,.5)
        result.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                      {'rankings':rankings,'changed_candidates':len(set(wide)-set(old))})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    stages=p.add_subparsers(dest='stage',required=True)
    setup, scoring=(stages.add_parser(name) for name in ('prepare','evaluate'))
    for entry in (setup,scoring):
        for k in ('manifest','reference','output'):
            entry.add_argument(f'--{k}',type=Path,required=True)
        entry.add_argument('--pointwise',type=Path,nargs='+',required=True)
    setup.add_argument('--initial-scan',type=Path,nargs='+',required=True)
    setup.add_argument('--missing-manifest',type=Path,required=True)
    for k in ('data-root','consensus','selection','reserve-ids'):
        scoring.add_argument(f'--{k}',type=Path,required=k in ('data-root','consensus'))
    scoring.add_argument('--evidence',type=Path,nargs='+',required=True)
    scoring.add_argument('--confirmation',action='store_true')
    args=p.parse_args()
    if args.output.exists() or (args.stage=='prepare' and args.missing_manifest.exists()):
        p.error('Use fresh output paths')
    if args.stage=='evaluate' and args.confirmation and not args.selection:
        p.error('Confirmation requires a frozen choice')
    result=prepare(args) if args.stage=='prepare' else evaluate(args,replay,ARMS)
    with args.output.open('x',encoding='utf8') as f:
        json.dump(result,f,separators=(',',':'))


if __name__=='__main__':
    main()
