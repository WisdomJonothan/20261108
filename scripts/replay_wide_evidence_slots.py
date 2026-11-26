"""Place new candidates into a fixed evidence budget before vote aggregation."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges
from eval_temporal_condition_intervention import evaluate
from eval_temporal_evidence_categories import ranking
from replay_anchor_fallback import index
from replay_consensus_head import head
from replay_evidence_ranking_consensus import reciprocal
from replay_feedback_admission import merge_proposal_grades
from replay_passage_admission import observations


ARMS = ('current', 'one_open_slot', 'two_open_slots')


def allocate(docs, challengers, old, new, cap):
    positions = (11, 13)
    incumbents = [docs[i] for i in positions]
    proposed = sorted(challengers, key=new.__getitem__, reverse=True)[:cap]
    scores = old | new
    winners = sorted(incumbents + proposed, key=scores.__getitem__, reverse=True)[:2]
    added = iter(d for d in winners if d not in incumbents)
    allocated = docs.copy()
    for i in positions:
        if docs[i] not in winners:
            allocated[i] = next(added)
    return allocated


def fusion(docs, grades, prior):
    reasoned = ranking(docs, [grades[d]['reasoned'] for d in docs])
    verified = ranking(docs, [grades[d]['reasoned' if grades[d]['verified'] else 'direct']
                              for d in docs])
    order = reciprocal([prior['current'], reasoned, verified, prior['pointwise']], 60)
    return head(order, prior['pointwise'], grades, True, .5)


def replay(args):
    shortlist = index({'per_query':load(args.manifest)['queries']})
    pools, consensus, frozen = (index(load(p)) for p in (args.pools, args.consensus, args.reference))
    old, proposals = observations(args.old_evidence), merge_proposal_grades(args.new_evidence)
    pointwise, new = (candidate_judges(paths)[0]['pretrained']
                      for paths in (args.old_pointwise,args.new_pointwise))
    if any(source.keys() != shortlist.keys()
           for source in (pools, consensus, frozen, old, proposals, pointwise, new)):
        raise ValueError('The comparison requires the same query IDs in every score source')
    output = []
    for key, row in shortlist.items():
        docs = row['docs']
        pool = pools[key]
        offers = pool['offers'][args.source]
        if (len(docs) != 14 or len(set(docs)) != 14 or not set(docs) <= set(pool['original'])
                or len(set(offers)) != 4 or not set(offers).isdisjoint(pool['original'])):
            raise ValueError(f'Unexpected candidate identities for {key}')
        if old[key].keys() != set(docs) or not set(offers) <= proposals[key].keys():
            raise ValueError(f'Missing fixed-document evidence grades for {key}')
        previous, upcoming = pointwise[key]['scores'], new[key]['scores']
        if not set(docs) <= previous.keys() or not set(offers) <= upcoming.keys():
            raise ValueError(f'Missing pointwise grades for {key}')
        current = frozen[key]['rankings']['verified_0.5']
        votes = consensus[key]['rankings']
        if fusion(docs, old[key], votes) != current or pool['prior_ten'] != current:
            raise ValueError(f'The unchanged fourteen documents must match the frozen ordering for {key}')
        orders = {'current': current}
        allocations = {'current': docs}
        estimates = old[key] | {d:proposals[key][d] for d in offers}
        for cap, label in enumerate(ARMS[1:],1):
            allocated = allocate(docs, offers, previous, upcoming, cap)
            orders[label] = current if docs == allocated else fusion(allocated, estimates, votes)
            allocations[label] = allocated
        output.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                      {'rankings':orders,'allocations':allocations})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','pools','consensus','reference','data-root','output','selection','reserve-ids'):
        parser.add_argument(f'--{name}', type=Path, required=name not in ('selection','reserve-ids'))
    for name in ('old-pointwise','new-pointwise','old-evidence','new-evidence'):
        parser.add_argument(f'--{name}', type=Path, nargs='+', required=True)
    parser.add_argument('--source', default='diver')
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if args.output.exists() or (args.confirmation and not args.selection):
        parser.error('Use a fresh output and supply a frozen choice for confirmation')
    report = evaluate(args, replay, ARMS)
    with args.output.open('x', encoding='utf8') as file:
        json.dump(report,file,separators=(',',':'))


if __name__ == '__main__':
    main()
