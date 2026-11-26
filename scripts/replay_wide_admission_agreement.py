"""Use agreement across two independently trained pair scorers for one admission."""
import argparse
import json
from math import log2
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges
from eval_temporal_condition_intervention import evaluate
from replay_anchor_fallback import index
from replay_feedback_admission import merge_proposal_grades, replay as original_replay
from replay_passage_admission import observations
from retrieve_union_with_pointwise_scan import REVISION, SOURCES


ARMS = ('current', 'wide_both_cross_encoders')
SOURCE = SOURCES[1]


def agreement(prior, old, new, candidates, senior, junior):
    best, output = 0., prior
    for i, incumbent in enumerate(prior[2:], 2):
        old_grade = old[incumbent]
        threshold = old_grade['reasoned' if old_grade['verified'] else 'direct']
        for contender in candidates:
            grade = new[contender]
            gain = (grade['reasoned'] - threshold) / log2(i+2)
            if (grade['verified'] and grade['reasoned'] >= 2 and gain > best
                    and senior[contender] > senior[incumbent]
                    and junior[contender] > junior[incumbent]):
                best = gain
                output = prior[:i] + [contender] + prior[i+1:]
    return output


def replay(args):
    original = index({'per_query': original_replay(args)})
    rows = index({'per_query':load(args.manifest)['queries']})
    scanned, identity, _ = candidate_judges(args.initial_scan)
    scan = scanned['pretrained']
    old, new = (candidate_judges(paths)[0]['pretrained']
                for paths in (args.old_pointwise, args.new_pointwise))
    pre, post = observations(args.old_evidence), merge_proposal_grades(args.new_evidence)
    if scan.keys() != rows.keys() or identity['pretrained']['revision'] != REVISION:
        raise ValueError('The initially scored union must cover every query')
    output = []
    for key, row in rows.items():
        current = original[key]['rankings']['current']
        candidates = row['proposals'][SOURCE]
        large = old[key]['scores'] | new[key]['scores']
        small = scan[key]['scores']
        if len(candidates) != 4 or not set(current+candidates) <= small.keys():
            raise ValueError(f'Missing comparable broad-scan scores for {key}')
        choice = agreement(current, pre[key], post[key], candidates, large, small)
        output.append({k:row[k] for k in ('qid','domain','split','cohort','actual_reference')} |
                      {'rankings': {'current':current, 'fixed_half_point':original[key]['rankings'][SOURCE],
                                    'wide_both_cross_encoders':choice}})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','reference','data-root','output','selection','reserve-ids'):
        p.add_argument(f'--{k}',type=Path,required=k not in ('selection','reserve-ids'))
    for k in ('old-pointwise','new-pointwise','old-evidence','new-evidence','initial-scan'):
        p.add_argument(f'--{k}',type=Path,nargs='+',required=True)
    p.add_argument('--confirmation',action='store_true')
    args=p.parse_args()
    if args.output.exists() or (args.confirmation and not args.selection):
        p.error('Use a fresh output and supply a frozen choice for confirmation')
    report=evaluate(args,replay,ARMS)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(report,file,separators=(',',':'))


if __name__=='__main__':
    main()
