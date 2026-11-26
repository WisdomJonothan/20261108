"""Build label-matched adapter groups with positives from two retrieval depths."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from coverage_cf_bound import read_qrels, read_steps
from lib_reteco import load_examples, read_trec_run
from run_qwen3_rerank import PREFIX, SUFFIX
from score_specialist_shortlists import TEMPORAL_INSTRUCT, evidence


def key(row):
    return row['domain'], row['qid']


def select(rows, limit, tag):
    identity = lambda row: hashlib.sha256(('|'.join((tag, *key(row)))).encode()).digest()
    return sorted(rows, key=identity)[:limit]


def reserve_ids(source, excluded):
    by_domain = defaultdict(list)
    for row in source:
        if row['split'] == 'train' and key(row) not in excluded:
            by_domain[row['domain']].append(row)
    reserved = [key(row) for d, rows in sorted(by_domain.items())
                for row in select(rows, 60, 'wide-evidence-additional-reserve-20261126')]
    if len(reserved) != len(set(reserved)) or not reserved:
        raise ValueError('A fresh reserve requires distinct training queries')
    return reserved


def build_domain(directory, args, excluded, archive_queries):
    domain = directory.name
    ids, texts = load_examples(directory, 'train')
    gold = read_qrels(directory/'qrels_train.txt')
    steps = (read_steps(directory/'qrels_steps_train.txt')
             if (directory/'qrels_steps_train.txt').exists() else {})
    dense = read_trec_run(args.dense_runs/f'dense_mxbai_{domain}_train.trec')
    bge = read_trec_run(args.bge_runs/f'dense_bge_{domain}_train.trec')
    common, matched = [], []
    for qid, fallback in zip(ids,texts,strict=True):
        if (domain,qid) in excluded:
            continue
        first, second = ([doc for _, _, doc in run[qid][:200]] for run in (dense,bge))
        near = [doc for doc in first[:60] if doc in gold[qid]]
        if not near:
            continue
        interleaved = list(dict.fromkeys(doc for pair in zip_longest(first,second)
                                        for doc in pair if doc is not None))
        broad = [doc for doc in interleaved if doc in gold[qid] and doc not in first[:60]]
        relevant_steps = set().union(*steps.get(qid,{}).values())
        known = gold[qid] | relevant_steps
        ordinary = [doc for doc in first[:60] if doc not in known][:2]
        fresh = [doc for doc in interleaved if doc not in known and doc not in first[:60]][:2]
        if len(ordinary)!=2 or len(fresh)!=2:
            continue
        row = {'domain':domain,'qid':qid,'query':archive_queries.get((domain,qid),fallback),
               'near':near[0],'negative':ordinary+fresh}
        common.append(row)
        if broad:
            matched.append(row | {'wide':broad[0]})
    return (select(common,140,'wide-evidence-common-20261126'),
            select(matched,70,'wide-evidence-matched-20261126'))


def make_records(rows, matched, corpus, tokenizer, prefix, suffix, max_length):
    arms = {'control':[], 'wide':[]}
    clipped = Counter()
    room = max_length-len(prefix)-len(suffix)
    for grade, items in (('common',rows),('matched',matched)):
        for row in items:
            question = row['query']
            query_ids = tokenizer.encode(question,add_special_tokens=False)
            cap = room-300
            if len(query_ids)>cap:
                half = cap//2
                question = tokenizer.decode(query_ids[:half]) + ' … ' + tokenizer.decode(query_ids[-(half-8):])
            header = f'<Instruct>: {TEMPORAL_INSTRUCT}\n<Query>: {question}\n<Document>: '
            if len(tokenizer.encode(header,add_special_tokens=False)) >= room:
                raise ValueError(f'Query header exceeds the input limit for {key(row)}')
            vocabulary = terms(row['query'])
            choices = [row['near'],*row['negative']]
            if grade == 'matched':
                choices.append(row['wide'])
            prompt = [header+evidence(corpus[d],vocabulary,'dispersed') for d in choices]
            ids = tokenizer(prompt,add_special_tokens=False)['input_ids']
            for name in arms:
                members = list(range(5))
                if name == 'wide' and grade == 'matched':
                    members[0]=5
                docs = [choices[i] for i in members]
                tokens = [prefix+ids[i][:room]+suffix for i in members]
                assert len(set(docs))==5
                clipped[name]+=sum(len(ids[i])>room for i in members)
                arms[name].append({'domain':row['domain'],'qid':row['qid'],'group':grade,
                                   'docs':docs,'labels':[1,0,0,0,0],'tokens':tokens})
    return arms, clipped


def prepare(args):
    from transformers import AutoTokenizer

    old = {key(r) for path in args.exclude for r in load(path)['queries']}
    if len(old)!=532:
        raise ValueError('Expected the three previous evaluation populations')
    reference = load(args.reserve_source)['queries']
    reserve = reserve_ids(reference,old)
    excluded = old | set(reserve)
    archive_queries = {key(r):r['query'] for r in reference}
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    prefix,suffix = [tokenizer.encode(text,add_special_tokens=False) for text in (PREFIX,SUFFIX)]
    records = {'control':[],'wide':[]}
    common_count, matched_count, clipped = Counter(),Counter(),Counter()
    for directory in sorted(d for d in args.data_root.iterdir() if (d/'qrels_train.txt').is_file()):
        common,matched = build_domain(directory,args,excluded,archive_queries)
        if not common:
            continue
        needed = {d for row in common+matched for d in [row['near'],*row['negative'],row.get('wide')] if d}
        corpus = read_documents(directory/'documents.jsonl',needed)
        groups, overflow = make_records(common,matched,corpus,tokenizer,prefix,suffix,args.max_length)
        for name in records:
            records[name].extend(groups[name])
        common_count[directory.name]=len(common)
        matched_count[directory.name]=len(matched)
        clipped.update(overflow)
        print(directory.name,len(common),len(matched),flush=True)
    paired = zip(records['control'],records['wide'],strict=True)
    for left,right in paired:
        assert key(left)==key(right) and key(left) not in excluded and left['group']==right['group']
        assert left['labels']==right['labels'] and left['docs'][1:]==right['docs'][1:]
        assert left['tokens'][1:]==right['tokens'][1:]
        assert (left['docs'][0]==right['docs'][0])==(left['group']=='common')
    counts = Counter(r['domain'] for r in records['control'])
    if not counts or not sum(matched_count.values()):
        raise ValueError('No matched adapter data was assembled')
    args.output.mkdir(parents=True)
    summary = {'counts':counts,'common_by_domain':common_count,'matched_by_domain':matched_count,
               'clipped_documents':clipped,'excluded_prior_query_count':len(old),
               'extra_reserve_count':len(reserve),'max_length':args.max_length,
               'model_type':load(args.model/'config.json')['model_type'],'pad':tokenizer.pad_token_id}
    outputs = {'reserve-ids':{'queries':reserve},'summary':summary}
    for name, groups in records.items():
        outputs[name]=summary | {'groups':groups}
    for name, data in outputs.items():
        with (args.output/f'{name}.json').open('x',encoding='utf8') as f:
            json.dump(data,f,separators=(',',':'))
    print('Matched preparation:',summary,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data-root','dense-runs','bge-runs','model','output','reserve-source'):
        p.add_argument(f'--{k}',type=Path,required=True)
    p.add_argument('--exclude',type=Path,nargs=3,required=True)
    p.add_argument('--max-length',type=int,default=1024)
    args=p.parse_args()
    if args.output.exists() or args.max_length<512:
        p.error('Use a fresh output directory and select at least 512 tokens')
    prepare(args)


if __name__=='__main__':
    main()
