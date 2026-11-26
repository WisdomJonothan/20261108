"""Compare and replay positive-sampling adapters on a fixed expanded candidate pool."""
import argparse
import json
from pathlib import Path
from statistics import mean

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import POLICIES, blend, candidate_judges
from eval_temporal_condition_intervention import evaluate
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from replay_passage_admission import observations


ARMS = ('current', 'wide_positive_adapter')
ALL_JUDGES = ('frozen_8b', 'near_positive_adapter', 'wide_positive_adapter')


def prepare(args):
    original = index({'per_query':load(args.manifest)['queries']})
    proposals = index({'per_query':load(args.scopes)['queries']})
    if original.keys()!=proposals.keys():
        raise ValueError('The scored question IDs must match the wider sample')
    rows = []
    for k,r in original.items():
        p = proposals[k]
        if r['docs']!=p['original'] or len(p['wide'])!=60:
            raise ValueError(f'The saved candidate sets differ for {k}')
        prior = p['original'][:30]
        extra = [doc for doc in p['wide'] if doc not in prior][:10]
        docs = prior[:20] + extra
        if len(docs)!=30 or len(set(docs))!=30:
            raise ValueError(f'Expected a fixed thirty-document comparison for {k}')
        rows.append(r | {'docs':docs, 'original30':prior})
    return {'queries':rows}


def checkpoint(paths):
    scores, metadata, _ = candidate_judges(paths)
    if scores.keys()!={'epoch1'}:
        raise ValueError('Expected the preselected one-epoch checkpoint')
    return scores['epoch1'],metadata['epoch1']


def replay(args):
    manifest = index({'per_query':load(args.manifest)['queries']})
    votes, reference = (index(load(p)) for p in (args.consensus,args.reference))
    base = candidate_judges(args.base)[0]['pretrained']
    near,near_id = checkpoint(args.near)
    wide,wide_id = checkpoint(args.wide)
    if near_id['sha256']==wide_id['sha256']:
        raise ValueError('Expected distinct positive-sampling checkpoints')
    comparable = ('epoch','groups','parameters','seed','lr','batch_groups','accumulate','world_size')
    if any(near_id['training'][k]!=wide_id['training'][k] for k in comparable):
        raise ValueError('Checkpoints have different training settings')
    evidence = observations(args.evidence)
    if any(source.keys()!=manifest.keys() for source in (votes,reference,base,near,wide,evidence)):
        raise ValueError('Every comparison requires the same query IDs')
    result = []
    for k,row in manifest.items():
        docs = row['docs']
        measured = [p[k]['scores'] for p in (base,near,wide)]
        if any(x.keys()!=set(docs) for x in measured[1:]) or not set(docs+row['original30'])<=measured[0].keys():
            raise ValueError(f'Model scores must cover the same candidate identities for {k}')
        rankings = votes[k]['rankings']
        frozen = reference[k]['rankings']['verified_0.5']
        prior_votes = [rankings[n] for n in ('current','reasoned','verified')]
        old_point = rankings['pointwise']
        settings = POLICIES['adaptive30_free']
        expected = blend(row['original30'], {}, measured[0], settings)
        if expected!=old_point or head(reciprocal(prior_votes+[expected],60),old_point,evidence[k],True,.5)!=frozen:
            raise ValueError(f'The original unadapted vote is different for {k}')
        order = {'current':frozen}
        for arm,scores in zip(ALL_JUDGES,measured,strict=True):
            pointwise = blend(docs,{},scores,settings)
            order[arm] = head(reciprocal(prior_votes+[pointwise],60),old_point,evidence[k],True,.5)
            order[arm+'/standalone'] = sorted(docs,key=scores.__getitem__,reverse=True)[:10]
        result.append({field:row[field] for field in ('qid','domain','cohort','split','actual_reference')} |
                      {'rankings':order})
    return result


def matched_diagnostics(report):
    rows = report['per_query']
    if 'near_positive_adapter' not in rows[0]['scores']:
        return
    paired = [(r['domain'],r['scores']['wide_positive_adapter']-r['scores']['near_positive_adapter'])
              for r in rows]
    effect = paired_delta(paired)
    by_domain = effect['per_domain']
    report['wide_minus_near'] = effect
    report['wide_minus_near_minimum_leave_one_domain_out'] = min(
        mean(value for d,value in by_domain.items() if d!=omit) for omit in by_domain)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest='stage',required=True)
    setup,scoring = (stages.add_parser(name) for name in ('prepare','evaluate'))
    for entry in (setup,scoring):
        for name in ('manifest','output'):
            entry.add_argument(f'--{name}',type=Path,required=True)
    setup.add_argument('--scopes',type=Path,required=True)
    for name in ('reference','consensus','data-root','selection','reserve-ids'):
        scoring.add_argument(f'--{name}',type=Path,required=name not in ('selection','reserve-ids'))
    for name in ('base','near','wide','evidence'):
        scoring.add_argument(f'--{name}',type=Path,nargs='+',required=True)
    scoring.add_argument('--confirmation',action='store_true')
    args=p.parse_args()
    if args.output.exists() or (args.stage=='evaluate' and args.confirmation and not args.selection):
        p.error('Use a fresh output and a frozen rule on confirmation')
    if args.stage=='prepare':
        result=prepare(args)
    else:
        result=evaluate(args,replay,ARMS)
        matched_diagnostics(result)
    with args.output.open('x',encoding='utf8') as out:
        json.dump(result,out,separators=(',',':'))


if __name__=='__main__':
    main()
