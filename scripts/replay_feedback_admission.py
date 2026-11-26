"""Check one evidence-confirmed admission from a fresh retrieval view."""
import argparse
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges
from eval_temporal_condition_intervention import evaluate
from replay_anchor_fallback import index
from replay_passage_admission import observations, replacement
from retrieve_after_evidence_review import SOURCES


def prepare(args):
    queries = index({'per_query':load(args.manifest)['queries']})
    pools = load(args.pools)
    candidates = index(pools)
    sources = tuple(pools.get('sources',SOURCES))
    if candidates.keys() != queries.keys():
        raise ValueError('Evidence feedback and the frozen reference use different query IDs')
    output = []
    for key,row in queries.items():
        source = candidates[key]
        arms = source['offers']
        if source['original'] != row['docs'] or tuple(arms) != sources:
            raise ValueError(f'Unexpected candidate protocol for {key}')
        if any(len(docs)!=len(set(docs)) or len(docs)!=4 or set(docs)&set(row['docs']) for docs in arms.values()):
            raise ValueError(f'Each source must supply four previously unseen candidates for {key}')
        docs = list(dict.fromkeys(d for alternatives in arms.values() for d in alternatives))
        output.append(row | {'docs':docs, 'offers':arms})
    return {'queries':output}


def narrow(args):
    predictions = candidate_judges(args.pointwise)[0]['pretrained']
    rows = load(args.manifest)['queries']
    if {(r['domain'],r['qid']) for r in rows} != predictions.keys():
        raise ValueError('Missing pointwise scores for some candidate pools')
    output = []
    for row in rows:
        key = row['domain'],row['qid']
        scores = predictions[key]['scores']
        if scores.keys()!=set(row['docs']):
            raise ValueError(f'Mismatched candidate score identities for {key}')
        proposals = {arm:[max(options,key=scores.__getitem__)] for arm,options in row['offers'].items()}
        docs = list(dict.fromkeys(d for chosen in proposals.values() for d in chosen))
        output.append(row | {'docs':docs, 'proposals':proposals})
    return {'queries':output}


def replay(args):
    candidates = index({'per_query':load(args.manifest)['queries']})
    references = index(load(args.reference))
    old, new = (observations(paths) for paths in (args.old_evidence,args.new_evidence))
    previous, following = (candidate_judges(paths)[0]['pretrained']
                           for paths in (args.old_pointwise,args.new_pointwise))
    if not candidates.keys()==references.keys()==old.keys()==new.keys()==previous.keys()==following.keys():
        raise ValueError('The evidence and pointwise scorers must cover the same queries')
    rows = []
    for key,row in candidates.items():
        prior = references[key]['rankings']['verified_0.5']
        left,right = previous[key]['scores'], following[key]['scores']
        if new[key].keys()!=set(row['docs']) or not set(row['docs'])<=right.keys() or not set(prior)<=left.keys():
            raise ValueError(f'Unexpected document judgments for {key}')
        rankings = {'current':prior}
        for arm, allowed in row['proposals'].items():
            rankings[arm] = replacement(prior,old[key],new[key],left,right,allowed,2)
        rows.append({k:row[k] for k in ('qid','domain','split','cohort','actual_reference')} |
                    {'rankings':rankings})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='mode',required=True)
    first, second, scoring = (stages.add_parser(name) for name in ('prepare','narrow','evaluate'))
    for stage in (first,second,scoring):
        for key in ('manifest','output'):
            stage.add_argument(f'--{key}',type=Path,required=True)
    first.add_argument('--pools',type=Path,required=True)
    second.add_argument('--pointwise',type=Path,required=True,nargs='+')
    for key in ('reference','data-root','selection','reserve-ids'):
        scoring.add_argument(f'--{key}',type=Path,required=key not in ('selection','reserve-ids'))
    for key in ('old-pointwise','new-pointwise','old-evidence','new-evidence'):
        scoring.add_argument(f'--{key}',type=Path,required=True,nargs='+')
    scoring.add_argument('--confirmation',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    if args.mode=='evaluate' and args.confirmation and not args.selection:
        parser.error('Confirmation uses a previously selected rule')
    if args.mode=='evaluate':
        choices = tuple(load(args.manifest)['queries'][0]['proposals'])
        output = evaluate(args,replay,('current', *choices))
    else:
        output = (prepare if args.mode=='prepare' else narrow)(args)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(output,file,separators=(',',':'))


if __name__=='__main__':
    main()
