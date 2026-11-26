"""Build matched training groups for contrastive attribution to literal question clauses."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from math import sqrt
from pathlib import Path
import re

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from coverage_cf_bound import read_qrels
from run_qwen3_rerank import PREFIX, SUFFIX
from score_specialist_shortlists import TEMPORAL_INSTRUCT, evidence


STOP = set(
    'the and with where in what which who are how can that does do or of is this about for a to '
    'before any on it so if could would provide from there find look search using use data '
    'investigate study examine information source sources analyze identify describe mention '
    'documentation documented answer compare related articles relevant others only discuss'.split()
)
GUIDE = ' Score evidence for the highlighted request, with the entities and time context from the full query.'


def vocabulary(text):
    text = re.sub(r'<[^>]+>', ' ', text.lower())
    return {w for w in re.findall(r'[a-z0-9]+', text) if len(w) > 2 and w not in STOP}


def literal_requests(question):
    selected, seen = [], []
    for s in question.split('?')[:-1]:
        text = re.split(r'(?<=[.!;])\s+', s.strip())[-1].strip()
        words = vocabulary(text)
        if len(words) >= 3 and not any(len(words & old)/len(words | old) > .8 for old in seen):
            selected.append(text + '?')
            seen.append(words)
    return selected[:4]


def mapped_request(step, candidates):
    expr = vocabulary(step['step_instruction'])
    similarity = []
    for i, words in enumerate(candidates):
        overlap = len(words & expr)
        similarity.append((overlap/sqrt(max(1, len(words)*len(expr))), i, overlap))
    first, second = sorted(similarity, reverse=True)[:2]
    if first[0] >= .17 and first[0] - second[0] >= .055 and first[2] >= 2:
        return first[1]
    return None


def step_pairs(args, blocked):
    for directory in sorted(args.data_root.iterdir()):
        source = directory/'steps_train.jsonl'
        if not source.is_file():
            continue
        gold = read_qrels(directory/'qrels_train.txt')
        with source.open(encoding='utf-8') as file:
            for line in file:
                row = json.loads(line)
                qid = row['id']
                if (directory.name, qid) in blocked:
                    continue
                phrases = literal_requests(row['query'])
                if len(phrases) < 2:
                    continue
                vocab = [vocabulary(clause) for clause in phrases]
                steps = [(step, mapped_request(step, vocab)) for step in row['steps']]
                for (left, a), (right, b) in combinations(steps, 2):
                    if a is None or b is None or a == b:
                        continue
                    lset, rset = set(left['gold_ids']), set(right['gold_ids'])
                    ldocs = (lset-rset) & gold[qid]
                    rdocs = (rset-lset) & gold[qid]
                    if ldocs and rdocs:
                        choose = lambda docs: min(docs, key=lambda d:sha256(f'{qid}|{d}'.encode()).digest())
                        yield {'domain':directory.name, 'qid':qid, 'query':row['query'],
                               'steps':[left['step_id'],right['step_id']], 'positions':[a,b],
                               'focus':[phrases[a],phrases[b]], 'docs':[choose(ldocs),choose(rdocs)]}


def full_groups(prepared, blocked):
    result = []
    for row in prepared['groups']:
        if (row['domain'], row['qid']) in blocked:
            continue
        positives = [i for i, label in enumerate(row['labels']) if label][:2]
        negatives = [i for i, label in enumerate(row['labels']) if not label][:4]
        selected = positives+negatives
        if positives and negatives:
            result.append(row | {name:[row[name][i] for i in selected]
                                 for name in ('tokens','labels','docs')})
    return result


def pack_inputs(query, focus, excerpts, tokenizer, tokens, max_length):
    prefix, suffix = tokens
    room = max_length-len(prefix)-len(suffix)
    original = tokenizer.encode(query, add_special_tokens=False)
    extra = f'\nHighlighted request: {focus}' if focus else ''
    guide = GUIDE if extra else ''
    extra_ids = tokenizer.encode(extra, add_special_tokens=False)
    capacity = room-300-len(extra_ids)
    if capacity < 128:
        raise ValueError('The highlighted request exceeds the available prompt budget')
    rendered = query
    if len(original) > capacity:
        left = capacity//2
        rendered = tokenizer.decode(original[:left]) + ' … ' + tokenizer.decode(original[-(capacity-left-8):])
    header = f'<Instruct>: {TEMPORAL_INSTRUCT}{guide}\n<Query>: {rendered}{extra}\n<Document>: '
    body = tokenizer([header+x for x in excerpts], add_special_tokens=False)['input_ids']
    return [prefix+x[:room]+suffix for x in body]


def pair_inputs(row, texts, tokenizer, tokens, max_length, kind):
    query = row['query']
    original_vocabulary = terms(query)
    excerpts = [evidence(texts[d], original_vocabulary, 'dispersed') for d in row['docs']]
    out = []
    for i, request in enumerate(row['focus']):
        focus = request if kind == 'contrast' else None
        labels = [int(j==i) for j in range(2)] if kind == 'contrast' else [1,1]
        out.append({'domain':row['domain'], 'qid':f'{row["qid"]}/{row["steps"][0]}/{row["steps"][1]}/{i}',
                    'docs':row['docs'], 'labels':labels,
                    'tokens':pack_inputs(query,focus,excerpts,tokenizer,tokens,max_length)})
    return out


def read_blocklist(args):
    held = {(r['domain'],r['qid']) for r in load(args.holdout)['queries']}
    confirmation = {(d,q) for d,q,_ in load(args.confirmation)}
    reserve = {tuple(x) for x in load(args.reserve)['queries']}
    all_ids = held | confirmation | reserve
    if (len(held),len(confirmation),len(reserve),len(all_ids)) != (186,155,191,532):
        raise ValueError('The three predeclared training evaluation populations must be disjoint')
    return all_ids


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','model','base-input','holdout','confirmation','reserve','output'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--max-length',type=int,default=1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    blocked = read_blocklist(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = load(args.model/'config.json')
    base = load(args.base_input)
    if base['model_type'] != config['model_type'] or base['pad'] != tokenizer.pad_token_id:
        raise ValueError('The frozen training inputs use different model tokenization')
    tokens = [tokenizer.encode(s,add_special_tokens=False) for s in (PREFIX,SUFFIX)]
    common = full_groups(base, blocked)
    identified = defaultdict(list)
    for item in step_pairs(args,blocked):
        identified[item['domain']].append(item)
    if len(common)==0 or len(identified)==0:
        raise ValueError('Expected full-query groups and distinct step pairs')
    by_arm = dict(full=common.copy(), contrast=common.copy())
    for domain, rows in sorted(identified.items()):
        texts = read_documents(args.data_root/domain/'documents.jsonl', {d for r in rows for d in r['docs']})
        for row in rows:
            for arm, groups in by_arm.items():
                groups.extend(pair_inputs(row,texts,tokenizer,tokens,args.max_length,arm))
        print(domain,len(rows),'matched contrast pairs',flush=True)
    args.output.mkdir(parents=True)
    for arm, groups in by_arm.items():
        with (args.output/f'{arm}.json').open('x') as f:
            json.dump({'groups':groups,'counts':dict(Counter(r['domain'] for r in groups)),
                       'model_type':config['model_type'],'pad':tokenizer.pad_token_id},
                      f,separators=(',',':'))
    ids = [{k:r[k] for k in ('domain','qid','steps','positions','docs')}
           for rows in identified.values() for r in rows]
    with (args.output/'pair-ids.json').open('x') as f:
        json.dump({'blocked':len(blocked),'shared_full_groups':len(common),'pairs':ids},f,separators=(',',':'))
    print(len(common),'shared full-query groups;',len(ids),'contrast pairs',flush=True)


if __name__=='__main__':
    main()
