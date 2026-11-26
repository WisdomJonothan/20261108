"""Account for the dependence between two votes from the same evidence grader."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_anchor_reference import load
from eval_temporal_condition_intervention import evaluate
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal


ARMS = ('current', 'group_mean', 'group_max', 'dependence_discount')
SOURCES = ('current', 'pointwise', 'reasoned', 'verified')


def contribution(order):
    return {doc:1/(60+rank) for rank,doc in enumerate(order,start=1)}


def combined_orders(lists):
    stable = list(dict.fromkeys(doc for i in (0, 2, 3, 1) for doc in lists[i]))
    first, point, reasoned, verified = map(contribution, lists)
    votes = [[part.get(d,0.) for d in stable] for part in (reasoned,verified)]
    # Both views share a checkpoint, query and passages. Use their vote correlation
    # as an observable proxy for repeated information within this specific query.
    rho = max(0., float(np.corrcoef(votes)[0,1]))
    first_scores = np.array([first.get(d,0.)+point.get(d,0.) for d in stable])
    support = np.array(votes)
    alternatives = {
        'group_mean': support.mean(axis=0),
        'group_max': support.max(axis=0),
        'dependence_discount': support.sum(axis=0)/(1.+rho),
    }
    return {name:[stable[i] for i in np.argsort(-(first_scores+bonus),kind='stable')[:10]]
            for name,bonus in alternatives.items()}, rho


def replay(args):
    queries = index({'per_query':load(args.manifest)['queries']})
    anchors, refs = (index(load(p)) for p in (args.consensus,args.reference))
    observations = index({'per_query':[r for p in args.evidence for r in load(p)['per_query']]})
    if not queries.keys()==anchors.keys()==refs.keys()==observations.keys():
        raise ValueError('Expected the same query list from each source')
    rows = []
    for key,row in queries.items():
        anchors_for_key = anchors[key]['rankings']
        lists = tuple(anchors_for_key[name] for name in SOURCES)
        grades = observations[key]['observations']
        start = refs[key]['rankings']['verified_0.5']
        initial_votes = tuple(anchors_for_key[name] for name in ('current','reasoned','verified','pointwise'))
        if head(reciprocal(initial_votes,60),lists[1],grades,True,.5)!=start:
            raise ValueError(f'Unexpected original consensus for {key}')
        predict, rho = combined_orders(lists)
        hypotheses = {name:head(docs,lists[1],grades,True,.5) for name,docs in predict.items()}
        rows.append({k:row[k] for k in ('qid','domain','split','cohort','actual_reference')} |
                    {'rankings':{'current':start}|hypotheses,'shared_evidence_correlation':rho})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','consensus','reference','data-root','output','selection','reserve-ids'):
        parser.add_argument(f'--{name}',type=Path,required=name not in ('selection','reserve-ids'))
    parser.add_argument('--evidence',type=Path,required=True,nargs='+')
    parser.add_argument('--confirmation',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    if args.confirmation and not args.selection:
        parser.error('Confirmation requires a fixed policy')
    result = evaluate(args,replay,ARMS)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(result,file,separators=(',',':'))


if __name__=='__main__':
    main()
