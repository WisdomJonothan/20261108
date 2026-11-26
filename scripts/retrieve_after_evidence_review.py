"""Choose one literal subquestion for a new search using the current evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index


SOURCES = ('whole', 'query_only', 'after_current_results')
INSTRUCTION = (
    'Choose the one numbered request that most needs a separate search to find supporting evidence. '
    'If retrieved excerpts are supplied, identify a required detail that those excerpts establish least clearly. '
    'Consider the time, event order and requested versions in the full question. '
    'Output only a listed request number.'
)


def query_view(query, fragment):
    if fragment is None:
        return query
    lead = ' '.join(query.splitlines()[0].split()[:32])
    return f'{lead}\n{fragment}'


def select_and_encode(args):
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from run_dense_generic import encode
    from audit_rejected_swap_evidence import read_documents
    from benchmark_pairwise_replacement_v2 import terms
    from prepare_step_contrast_adapter import literal_requests
    from lib_reteco import load_examples
    from retrieve_passage_evidence import PREFIX
    from score_specialist_shortlists import evidence

    rows = load(args.manifest)['queries'][args.shard_index::args.shards]
    references = index(load(args.reference))
    split, = {r['split'] for r in rows}
    originals = {d:dict(zip(*load_examples(args.data_root/d,split),strict=True))
                 for d in {r['domain'] for r in rows}}
    tokenizer = AutoTokenizer.from_pretrained(args.selector,padding_side='left',local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.selector,dtype=torch.bfloat16,
                                               local_files_only=True).to('cuda').eval()
    choices = [tokenizer.encode(s,add_special_tokens=False) for s in '1234']
    if any(len(c)!=1 for c in choices) or len({c[0] for c in choices})!=4:
        raise ValueError('Expected four distinct question-choice tokens')
    label_ids = [c[0] for c in choices]
    groups = defaultdict(list)
    for row in rows:
        groups[row['domain']].append(row)
    metadata, queries = [], []
    started = perf_counter()
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            anchors = {r['qid']:references[domain,r['qid']]['rankings']['verified_0.5'][:4] for r in group}
            docs = read_documents(args.data_root/domain/'documents.jsonl', {d for ids in anchors.values() for d in ids})
            prompts, active = [], {}
            for row in group:
                qid, query = row['qid'], row['query']
                requests = literal_requests(query)
                active[qid] = requests
                if len(requests)<2:
                    continue
                scored = terms(query)
                texts = [evidence(docs[d],scored,'temporal') for d in anchors[qid]]
                numbered = '\n'.join(f'{i}. {s}' for i,s in enumerate(requests,1))
                stem = f'{INSTRUCTION}\n\nQuestion\n{query}\n\nRequests\n{numbered}'
                context = '\n\nRetrieved excerpts\n'+'\n\n'.join(texts)
                for suffix in ('',context):
                    chat = tokenizer.apply_chat_template([{'role':'user','content':stem+suffix}],
                                                          tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    prompts.append(chat+'Request: ')
            values = []
            for start in range(0,len(prompts),args.batch_size):
                batch = tokenizer(prompts[start:start+args.batch_size],padding=True,
                                  add_special_tokens=False,return_tensors='pt').to(model.device)
                if batch.input_ids.shape[1]>4096:
                    raise ValueError(f'Expected four bounded evidence excerpts in {domain}')
                logits = model(**batch,use_cache=False,logits_to_keep=1).logits[:, -1, label_ids]
                values.extend(logits.float().cpu().tolist())
            cursor = 0
            for row in group:
                qid, query = row['qid'], row['query']
                requests = active[qid]
                selected = [None,None]
                if len(requests)>=2:
                    pair = np.array(values[cursor:cursor+2])[:, :len(requests)]
                    selected = pair.argmax(axis=1).tolist()
                    cursor += 2
                original = originals[domain][qid]
                views = [original, *[query_view(query, requests[i]) if i is not None else original
                                     for i in selected]]
                queries.extend(PREFIX+view for view in views)
                metadata.append({'domain':domain,'qid':qid,'request_count':len(requests),'selected':selected})
            if cursor!=len(values):
                raise ValueError(f'Unassigned selection scores in {domain}')
            print(domain,len(metadata),round(perf_counter()-started),flush=True)
    del tokenizer,model
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder,local_files_only=True)
    model = AutoModel.from_pretrained(args.encoder,local_files_only=True).to('cuda',dtype=torch.float16).eval()
    vectors = encode(queries,model,tokenizer,'cuda',16,512,'mean','local retrieval queries')
    info = {'split':split,'encoder':args.encoder.name,'rewriter':args.selector.name,
            'instruction':INSTRUCTION,'views':metadata}
    with args.output.open('xb') as file:
        np.savez_compressed(file,metadata=json.dumps(info),vectors=vectors.reshape(len(rows),3,-1))


def make_pools(args):
    from retrieve_temporal_query_views import read_views, retrieve

    manifest = index({'per_query':load(args.manifest)['queries']})
    references = index(load(args.reference))
    split, = {r['split'] for r in manifest.values()}
    views, models = read_views(args.vectors, split)
    if manifest.keys()!=views.keys() or manifest.keys()!=references.keys():
        raise ValueError('The original pipeline, candidate pool and query views must cover the same questions')
    domains = defaultdict(list)
    for (domain,_), row in sorted(manifest.items()):
        domains[domain].append(row)
    output = []
    for domain, group in sorted(domains.items()):
        with (args.data_root/domain/'documents.jsonl').open(encoding='utf8') as file:
            docids = [json.loads(line)['id'] for line in file if line.strip()]
        rankings = retrieve(docids,args.embeddings,group,views)
        for row,new_lists in zip(group,rankings,strict=True):
            key = domain,row['qid']
            original = set(row['docs'])
            if len(original)!=60:
                raise ValueError(f'Expected a fixed 60-document reference for {key}')
            offers = {arm:[doc for doc in ranked if doc not in original][:4]
                      for arm,ranked in zip(SOURCES,new_lists,strict=True)}
            if any(len(docs)!=4 for docs in offers.values()):
                raise ValueError(f'There must be four new candidates per source for {key}')
            start = references[key]['rankings']['verified_0.5']
            output.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                          {'original':row['docs'],'prior_ten':start,'offers':offers,'choices':views[key][0]})
        print(domain,len(output),flush=True)
    return {'models':models,'sources':list(SOURCES),'per_query':output}


def audit(args):
    pools = load(args.pools)
    rows = pools['per_query']
    if {r['split'] for r in rows}!={'train'}:
        raise ValueError('The candidate ceilings use only official training judgments')
    labels = {d:read_qrels(args.data_root/d/'qrels_train.txt') for d in {r['domain'] for r in rows}}
    for row in rows:
        gold = labels[row['domain']][row['qid']]
        baseline = row['prior_ten']
        row['current'] = ndcg10(baseline,gold)
        row['diagnostics'] = {}
        for arm,offers in row['offers'].items():
            new = set(offers) & gold
            allowed_edits = [ndcg10(baseline[:i]+[doc]+baseline[i+1:],gold) for i in range(2,10)
                             if baseline[i] not in gold for doc in new]
            row['diagnostics'][arm] = {'new_positives':len(new),'one_swap':max([row['current'],*allowed_edits])}
    summary = {}
    sources = tuple(pools.get('sources',SOURCES))
    if any(tuple(r['offers'])!=sources for r in rows):
        raise ValueError('Unexpected candidate sources in the diagnostic')
    for arm in sources:
        summary[arm] = {
            'queries_with_positives': sum(r['diagnostics'][arm]['new_positives']>0 for r in rows),
            'one_swap': macro((r['domain'],r['diagnostics'][arm]['one_swap']) for r in rows),
            'change': paired_delta([(r['domain'],r['diagnostics'][arm]['one_swap']-r['current']) for r in rows]),
        }
    print({arm:(round(v['one_swap'],6),v['queries_with_positives']) for arm,v in summary.items()},flush=True)
    return {'summary':summary,'per_query':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    stages=parser.add_subparsers(dest='stage',required=True)
    encode,select,diagnostic=(stages.add_parser(s) for s in ('encode','retrieve','audit'))
    for stage in (encode,select,diagnostic):
        for key in ('data-root','output'):
            stage.add_argument(f'--{key}',type=Path,required=True)
    for stage in (encode,select):
        for key in ('manifest','reference'):
            stage.add_argument(f'--{key}',type=Path,required=True)
    for key in ('encoder','selector'):
        encode.add_argument(f'--{key}',type=Path,required=True)
    for key,default in (('batch-size',8),('shards',1),('shard-index',0)):
        encode.add_argument(f'--{key}',type=int,default=default)
    select.add_argument('--vectors',type=Path,nargs='+',required=True)
    select.add_argument('--embeddings',type=Path,required=True)
    diagnostic.add_argument('--pools',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.stage=='encode':
        if not (0<=args.shard_index<args.shards and args.batch_size>0):
            parser.error('Check shard and batch scope')
        select_and_encode(args)
        return
    result=(make_pools if args.stage=='retrieve' else audit)(args)
    with args.output.open('x',encoding='utf8') as file:
        json.dump(result,file,separators=(',',':'))


if __name__=='__main__':
    main()
