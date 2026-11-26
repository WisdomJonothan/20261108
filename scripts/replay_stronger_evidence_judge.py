"""Measure a stronger evidence judge on the same documents and rubric."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_temporal_condition_intervention import evaluate
from eval_temporal_evidence_categories import ranking
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal


ARMS = ('current','teacher_replaces_evidence_votes','teacher_adds_evidence_vote')


def observations(paths):
    shards = [load(p) for p in paths]
    identities = {(p['model'],p['instruction'],p.get('evidence_mode','dispersed')) for p in shards}
    if len(identities)!=1:
        raise ValueError('Judgment shards need the same checkpoint and rubric')
    identity, = identities
    cached = index({'per_query':[r for part in shards for r in part['per_query']]})
    return identity, {key:item['observations'] for key,item in cached.items()}


def individual(docs, values, field):
    return sorted(docs,key=lambda d:-values[d][field])[:10]


def replay(args):
    source = index({'per_query':load(args.manifest)['queries']})
    consensus, reference = (index(load(p)) for p in (args.consensus,args.reference))
    old_id, old = observations(args.evidence)
    new_id, new = observations(args.teacher)
    if old_id[0]==new_id[0] or old_id[1:]!=new_id[1:]:
        raise ValueError('The new model should change with an identical grading procedure')
    if not source.keys()==consensus.keys()==reference.keys()==old.keys()==new.keys():
        raise ValueError('The comparison requires matched query IDs in all score sources')
    output = []
    for key,row in source.items():
        docs = row['docs']
        previous, teacher = old[key], new[key]
        if len(docs)!=14 or len(set(docs))!=14 or set(docs)!=previous.keys() or set(docs)!=teacher.keys():
            raise ValueError(f'Expected matched fourteen-document evidence for {key}')
        votes = consensus[key]['rankings']
        controls = tuple(votes[arm] for arm in ('current','reasoned','verified','pointwise'))
        base = reference[key]['rankings']['verified_0.5']
        if head(reciprocal(controls,60),votes['pointwise'],previous,True,.5)!=base:
            raise ValueError(f'Unexpected original consensus for {key}')
        evidence_vote = ranking(docs,[teacher[d]['reasoned'] for d in docs])
        checked = ranking(docs,[teacher[d]['reasoned'] if teacher[d]['verified'] else teacher[d]['direct']
                                for d in docs])
        alternatives = {
            'teacher_replaces_evidence_votes':(votes['current'],evidence_vote,checked,votes['pointwise']),
            'teacher_adds_evidence_vote':(*controls,checked),
        }
        orders = {'current':base}
        for arm, group in alternatives.items():
            orders[arm] = head(reciprocal(group,60),votes['pointwise'],previous,True,.5)
        for label, grades in (('original',previous),('teacher',teacher)):
            for field in ('direct','reasoned'):
                orders[f'{label}/{field}'] = individual(docs,grades,field)
        output.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                      {'rankings':orders,'verified_teacher_quotes':sum(obs['verified'] for obs in teacher.values())})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','consensus','reference','data-root','output','selection','reserve-ids'):
        parser.add_argument(f'--{k}',type=Path,required=k not in ('selection','reserve-ids'))
    for k in ('evidence','teacher'):
        parser.add_argument(f'--{k}',type=Path,nargs='+',required=True)
    parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.confirmation and not args.selection:
        parser.error('Confirmation needs a rule fixed earlier')
    report=evaluate(args,replay,ARMS)
    with args.output.open('x',encoding='utf8') as out:
        json.dump(report,out,separators=(',',':'))


if __name__=='__main__':
    main()
