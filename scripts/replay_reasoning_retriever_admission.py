"""Spend two fixed evidence slots on candidates favored by another retriever."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_diver_evidence_views import REVISION, evidence_views
from eval_temporal_condition_intervention import evaluate
from replay_anchor_cap import z
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index
from replay_facet_shortlist import evidence_rank, gather


ARMS = ('current', 'retriever_full_pool', 'retriever_clause_pool', 'retriever_marginal_pool')


def candidate_pools(args):
    fixed = index({'per_query':load(args.manifest)['queries']})
    wide = index({'per_query':load(args.wide)['queries']})
    if fixed.keys() != wide.keys():
        raise ValueError('The fixed evidence and wide retrieval pools must share query IDs')
    return fixed, wide


def tier_options(original, pool):
    if len(set(original)) != 14 or len(pool) != len(set(pool)) or len(pool) != 60:
        raise ValueError('Expected fourteen grading slots and sixty candidates')
    if not set(original) <= set(pool):
        raise ValueError('The wide pool must contain the original evidence shortlist')
    if not (set(original[10:12]) <= set(pool[:30]) and set(original[12:]) <= set(pool[30:])):
        raise ValueError('The four original challengers must come from their fixed retrieval tiers')
    return [(i, [original[i], *[doc for doc in tier if doc not in original]])
            for i, tier in ((11, pool[:30]), (13, pool[30:]))]


def plan(args):
    fixed, wide = candidate_pools(args)
    shards = [load(p) for p in args.scores]
    if {part['revision'] for part in shards}!={REVISION}:
        raise ValueError('Unexpected encoder for the matched candidate-selection trial')
    scores = index({'per_query':[r for shard in shards for r in shard['per_query']]})
    if not fixed.keys()==wide.keys()==scores.keys():
        raise ValueError('The evidence and wide retrieval pools must share query IDs')
    plans, unseen = [], []
    for key,row in fixed.items():
        pool = wide[key]['docs']
        original = row['docs']
        logits = scores[key]['scores']
        if set(pool) != logits.keys():
            raise ValueError(f'Unexpected score membership for {key}')
        alternatives = tier_options(original, pool)
        selections = {'current':original}
        positions = {doc:i for i,doc in enumerate(pool)}
        views = np.array([logits[doc] for doc in pool])
        clause_support = np.column_stack([z(column) for column in views[:,1:].T]) if views.shape[1] > 1 else None
        for arm in ARMS[1:]:
            docs = original.copy()
            retained = original[:11] + original[12:13]
            for i, options in alternatives:
                full, focused = evidence_views([logits[doc] for doc in options])
                if arm == 'retriever_marginal_pool' and clause_support is not None:
                    covered = clause_support[[positions[doc] for doc in retained]].max(axis=0)
                    offered = clause_support[[positions[doc] for doc in options]]
                    gains = np.maximum(0., (offered-covered).max(axis=1))
                    score = .5 * (full + z(gains))
                elif arm == 'retriever_clause_pool':
                    score = focused
                else:
                    score = full
                docs[i] = options[int(score.argmax())]
                retained.append(docs[i])
            selections[arm] = docs
        added = list(dict.fromkeys(d for policy in ARMS[1:] for d in selections[policy] if d not in original))
        plans.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                     {'shortlists':selections,'extra_docs':added})
        if added:
            unseen.append(row | {'docs':added})
    return {'per_query':plans,'queries':unseen}


def oracle(args):
    fixed, wide = candidate_pools(args)
    if {row['split'] for row in fixed.values()} != {'train'}:
        raise ValueError('This diagnostic uses the training-side candidate pools')
    qrels = {d:read_qrels(args.data_root/d/'qrels_train.txt') for d,_ in fixed}
    rows = []
    for (domain,qid), row in fixed.items():
        original = row['docs']
        alternatives = tier_options(original, wide[domain,qid]['docs'])
        gold = qrels[domain][qid]
        retained = set(original)
        retained.difference_update(original[i] for i,_ in alternatives)
        ideal = retained & gold
        for _, options in alternatives:
            positive = next((doc for doc in options if doc in gold), None)
            if positive is not None:
                ideal.add(positive)
        frozen = ndcg10(sorted(set(original) & gold),gold)
        reachable = ndcg10(sorted(ideal),gold)
        rows.append({'domain':domain, 'qid':qid, 'current':frozen, 'reachable':reachable})
    summary = {k:macro((r['domain'],r[k]) for r in rows) for k in ('current','reachable')}
    summary['change'] = paired_delta([(r['domain'],r['reachable']-r['current']) for r in rows])
    print(summary, flush=True)
    return {'summary':summary,'per_query':rows}


def replay(args):
    plans = index(load(args.manifest))
    references, consensuses = (index(load(p)) for p in (args.reference,args.consensus))
    grades = gather(args.evidence)
    if not plans.keys()==references.keys()==consensuses.keys()==grades.keys():
        raise ValueError('The replay needs every original evidence assessment')
    output = []
    for key,row in plans.items():
        pools = row['shortlists']
        anchors = consensuses[key]['rankings']
        current = references[key]['rankings']['verified_0.5']
        if any(not set(p) <= grades[key].keys() for p in pools.values()):
            raise ValueError(f'Missing assessments for {key}')
        by_policy = {name:evidence_rank(docs,grades[key],anchors['current'],anchors['pointwise'])
                     for name,docs in pools.items()}
        if by_policy['current']!=current:
            raise ValueError(f'The original fixed production list differs for {key}')
        output.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                      {'rankings':by_policy, 'extra_docs':row['extra_docs']})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    management, diagnostic, scoring = (modes.add_parser(name) for name in ('plan','oracle','evaluate'))
    for stage in (management,diagnostic,scoring):
        for path in ('manifest','output'):
            stage.add_argument(f'--{path}',type=Path,required=True)
    for stage in (management,diagnostic):
        stage.add_argument('--wide',type=Path,required=True)
    diagnostic.add_argument('--data-root',type=Path,required=True)
    management.add_argument('--scores',type=Path,nargs='+',required=True)
    for path in ('data-root','reference','consensus','selection','reserve-ids'):
        scoring.add_argument(f'--{path}',type=Path,required=path not in ('selection','reserve-ids'))
    scoring.add_argument('--evidence',type=Path,nargs='+',required=True)
    scoring.add_argument('--confirmation',action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.mode=='evaluate' and args.confirmation and not args.selection:
        parser.error('Confirmation needs a frozen comparison')
    action = {'plan':plan, 'oracle':oracle, 'evaluate':lambda options:evaluate(options,replay,ARMS)}
    data = action[args.mode](args)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(data,file,separators=(',',':'))


if __name__=='__main__':
    main()
