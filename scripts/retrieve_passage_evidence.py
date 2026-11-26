"""Retrieve with local supporting spans from the same dense candidate pool."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import passages, terms
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_temporal_evidence_categories import question_facets
from lib_reteco import read_trec_run
from replay_anchor_cap import z
from replay_anchor_edit_scope import paired_delta


PREFIX = 'Represent this sentence for searching relevant passages: '
ARMS = ('current', 'dense_only', 'full_span', 'facet_span')


def windows(body, full, facets):
    options = passages(body, full)
    picks = [options[0]]
    covered = terms(picks[0])
    coded = [(window, terms(window)) for window in options[1:]]
    for _ in range(2):
        if not coded:
            break
        def benefit(item):
            observed = item[1]
            return max((len((observed-covered)&f)/max(1,len(f)) for f in facets),
                       default=len(observed & (full-covered))/max(1,len(full)))
        selected = max(coded, key=benefit)
        if benefit(selected) == 0:
            break
        picks.append(selected[0])
        covered.update(selected[1])
        coded.remove(selected)
    return list(dict.fromkeys(picks))


def encode_domain(group, sources, texts, encoder):
    tokens, spine, queries = {}, {}, []
    for row in group:
        doc_ids = sources[row['qid']]
        question = row['query']
        exact = question_facets(question)
        full = terms(question)
        facets = [terms(part) for part in exact]
        query_views = [question, *exact]
        for q in query_views:
            tokens.setdefault(PREFIX+q, len(tokens))
        queries.append([tokens[PREFIX+q] for q in query_views])
        for doc in doc_ids:
            choice = windows(texts[doc], full, facets)
            for span in choice:
                tokens.setdefault(span, len(tokens))
            spine[row['qid'], doc] = [tokens[span] for span in choice]
    vectors = encoder(list(tokens))
    return vectors, spine, queries


def replay(group, sources, texts, encoder):
    vectors, spine, queries = encode_domain(group, sources, texts, encoder)
    result = []
    for row, query in zip(group, queries, strict=True):
        qid, original = row['qid'], row['docs']
        prefix = original[:30]
        outside = [d for d in sources[qid] if d not in prefix]
        value = vectors[query]
        best = np.stack([(vectors[spine[qid,doc]] @ value.T).max(axis=0) for doc in outside])
        whole = z(best[:,0])
        if best.shape[1] > 1:
            views = np.column_stack([z(col) for col in best[:,1:].T])
            focused = .5 * (whole + z(views.max(axis=1)))
        else:
            focused = whole
        early = z(1 / np.log2(2 + np.arange(len(outside))))
        orders = {'current': original, 'dense_only': prefix+outside[:30]}
        for arm, local in (('full_span',whole), ('facet_span',focused)):
            order = np.argsort(-(.45*early + .55*local), kind='stable')[:30]
            orders[arm] = prefix+[outside[i] for i in order]
        result.append({k: row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                      {'pools':orders,'encoded_views':len(query),'outside_depth':len(outside)})
    return result


def score(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from run_dense_generic import encode

    source = load(args.manifest)['queries'][args.shard_index::args.shards]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True).to('cuda', dtype=torch.float16).eval()

    def encoder(texts):
        return encode(texts, model, tokenizer, 'cuda', args.batch_size, 512, 'mean', 'passage windows')

    groups = defaultdict(list)
    for row in source:
        groups[row['domain'],row['split']].append(row)
    rows = []
    for (domain,split), group in sorted(groups.items()):
        ranked = read_trec_run(args.runs/f'dense_mxbai_{domain}_{split}.trec')
        sources = {r['qid']:list(dict.fromkeys([d for _,_,d in ranked[r['qid']][:args.depth]]+r['docs']))
                   for r in group}
        texts = read_documents(args.data_root/domain/'documents.jsonl',
                               {d for docs in sources.values() for d in docs})
        rows.extend(replay(group,sources,texts,encoder))
        print(domain,len(rows),'/',len(source),flush=True)
    return {'model':args.model.name,'depth':args.depth,'per_query':rows}


def audit(args):
    records = [load(path) for path in args.scores]
    if len({(r['model'],r['depth']) for r in records}) != 1:
        raise ValueError('The shards must share the same scope and encoder')
    rows = [q for part in records for q in part['per_query']]
    split, = {r['split'] for r in rows}
    keys = {(r['domain'],r['qid']) for r in rows}
    if len(keys) != len(rows):
        raise ValueError('Some query scores are repeated')
    gold = {d:read_qrels(args.data_root/d/f'qrels_{split}.txt') for d,_ in keys}
    for r in rows:
        labels=gold[r['domain']][r['qid']]
        r['oracles'] = {arm:ndcg10(sorted(set(docs)&labels),labels) for arm,docs in r['pools'].items()}
        r['positives'] = {arm:len(set(docs)&labels) for arm,docs in r['pools'].items()}
    summary = {arm:{'oracle':macro((r['domain'],r['oracles'][arm]) for r in rows),
                    'empty_queries':sum(r['positives'][arm]==0 for r in rows),
                    'change':paired_delta([(r['domain'],r['oracles'][arm]-r['oracles']['current']) for r in rows])}
               for arm in ARMS}
    print({k:(v['oracle'],v['empty_queries']) for k,v in summary.items()},flush=True)
    return {'summary':summary,'per_query':rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage',required=True)
    scoring, auditing = (stages.add_parser(name) for name in ('score','audit'))
    for sub in (scoring,auditing):
        for name in ('data-root','output'):
            sub.add_argument(f'--{name}',type=Path,required=True)
    for name in ('runs','manifest','model'):
        scoring.add_argument(f'--{name}',type=Path,required=True)
    for name,value in (('depth',300),('batch-size',64),('shards',1),('shard-index',0)):
        scoring.add_argument(f'--{name}',type=int,default=value)
    auditing.add_argument('--scores',type=Path,nargs='+',required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.stage == 'score' and not (args.depth>=60 and 0<=args.shard_index<args.shards):
        parser.error('Check corpus depth and shard bounds')
    output = (score if args.stage=='score' else audit)(args)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(output,file,separators=(',',':'))


if __name__ == '__main__':
    main()
