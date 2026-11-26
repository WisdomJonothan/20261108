"""Compare two search depths for finding four evidence candidates with one scorer."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from compare_anchor_reference import load
from eval_qwen_reranker_adapter import candidate_judges
from lib_reteco import read_trec_run
from replay_anchor_fallback import index
from retrieve_union_with_reasoning_encoder import four_new, union_for


SOURCES = ('pointwise_depth50', 'pointwise_depth200')
REVISION = '0101299a7937a016848d8d55b092f6e31cfa312d'
DEPTHS = (50, 200)


def prepare(args):
    manifest = load(args.manifest)['queries']
    groups = defaultdict(list)
    for row in manifest:
        groups[row['domain'], row['split']].append(row)
    out = []
    for (domain, split), rows in sorted(groups.items()):
        first, second = (read_trec_run(root/f'dense_{tag}_{domain}_{split}.trec')
                         for root, tag in ((args.mxbai, 'mxbai'), (args.bge, 'bge')))
        for row in rows:
            parents, full = union_for(row, first, second, DEPTHS[1])
            prefix = list(dict.fromkeys(parents[0][:DEPTHS[0]] + parents[1][:DEPTHS[0]] + row['docs']))
            if not set(prefix) <= set(full):
                raise ValueError(f'Unexpected candidate nesting for {domain}/{row["qid"]}')
            out.append(row | {'docs':full, 'prefix':prefix, 'original':row['docs']})
    return {'queries':out}


def pools(args):
    artifact = load(args.manifest)
    predictions, identities, _ = candidate_judges(args.scores)
    identity = identities['pretrained']
    if identity['revision'] != REVISION:
        raise ValueError('Use the same pinned small scorer at both search depths')
    measured = predictions['pretrained']
    frozen = index(load(args.reference))
    query_keys = {(r['domain'], r['qid']) for r in artifact['queries']}
    if query_keys != frozen.keys() or query_keys != measured.keys():
        raise ValueError('Search, reference and grading must use the same queries')
    output = []
    for row in artifact['queries']:
        key = row['domain'], row['qid']
        scores = measured[key]['scores']
        if scores.keys() != set(row['docs']):
            raise ValueError(f'Incomplete wide-pool scores for {key}')
        original = set(row['original'])
        offers = {arm:four_new(sorted(row[field], key=scores.__getitem__, reverse=True), original)
                  for arm,field in zip(SOURCES, ('prefix','docs'), strict=True)}
        output.append({k:row[k] for k in ('domain','qid','cohort','split','actual_reference')} |
                      {'original':row['original'], 'prior_ten':frozen[key]['rankings']['verified_0.5'],
                       'union_count':len(row['docs']), 'prefix_count':len(row['prefix']), 'offers':offers})
    return {'models':{'model':identity['model'], 'revision':REVISION, 'source_depth':DEPTHS[1],
                      'prefix_depth':DEPTHS[0]}, 'sources':list(SOURCES), 'per_query':output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest='stage', required=True)
    setup, collect = (stages.add_parser(s) for s in ('prepare','pools'))
    for stage in (setup, collect):
        stage.add_argument('--manifest', type=Path, required=True)
        stage.add_argument('--output', type=Path, required=True)
    for name in ('mxbai','bge'):
        setup.add_argument(f'--{name}', type=Path, required=True)
    collect.add_argument('--reference', type=Path, required=True)
    collect.add_argument('--scores', type=Path, required=True, nargs='+')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output path')
    artifact = (prepare if args.stage=='prepare' else pools)(args)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(artifact,file,separators=(',',':'))


if __name__=='__main__':
    main()
