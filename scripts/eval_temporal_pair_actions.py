"""Evaluate tie-aware pairwise evidence decisions on fixed ranking proposals."""
import argparse
from collections import defaultdict
import hashlib
import json
from math import log2
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_anchor_dev import macro
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_anchor_reranker import write_checkpoint
from replay_anchor_edit_scope import paired_delta
from replay_anchor_fallback import index
from score_specialist_shortlists import evidence


INSTRUCTION = (
    'Choose which passage provides stronger evidence for answering the query. '
    'Prioritize a precise fact or explanation that the passage actually supports. '
    'Respect the time periods and chronological relations relevant to that fact. '
    'A passage can support one part of a complex query. Consider evidence quality, not length. '
    'Return A for the first passage, B for the second passage, or C when the evidence is tied '
    'or neither provides a clear answer. Output one letter.'
)
QUOTE_INSTRUCTION = (
    'Compare the evidence for answering the query and its time constraints. '
    'Each passage can support one part of the question. Quote the best directly usable words from '
    'each passage before deciding, using NONE when it lacks an answer. Then choose the passage with '
    'stronger support, or C for a tie or two weak passages. Output exactly three lines in this format: '
    'A evidence: <short exact quote or NONE>\nB evidence: <short exact quote or NONE>\nChoice: <A, B or C>'
)
POLICIES = {'current': ('none', 0., False)}
for kind in ('head', 'admission', 'both'):
    POLICIES[f'{kind}_agree'] = (kind, 0., True)
    POLICIES[f'{kind}_agree_margin20'] = (kind, .20, True)
POLICIES['both_mean_margin15'] = ('both', .15, False)


def texts_for_domain(rows, path):
    needed = {p[k] for row in rows for p in row['proposals'] for k in ('incumbent', 'challenger')}
    found = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            doc = json.loads(line)
            if doc['id'] in needed:
                found[doc['id']] = doc['content']
                if found.keys() == needed:
                    break
    if found.keys() != needed:
        raise ValueError(f'Missing proposal documents in {path}')
    return found


def bounded(tokenizer, text, limit):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    front = limit // 2
    return tokenizer.decode(ids[:front]) + ' … ' + tokenizer.decode(ids[-(limit-front-8):])


def grounded_choice(output, documents):
    choice = re.search(r'^\s*choice\s*:\s*([ABC])\b', output, re.IGNORECASE | re.MULTILINE)
    if not choice or choice[1].upper() == 'C':
        return [0., 0., 1.]
    letter = choice[1].upper()
    quote = re.search(r'^\s*' + letter + r'\s+evidence\s*:\s*([^\n]+)',
                      output, re.IGNORECASE | re.MULTILINE)
    if not quote:
        return [0., 0., 1.]
    words = re.findall(r'\b\w+\b', quote[1].casefold())
    source = re.findall(r'\b\w+\b', documents['AB'.index(letter)].casefold())
    if len(words) < 5 or f' {" ".join(words)} ' not in f' {" ".join(source)} ':
        return [0., 0., 1.]
    return [1., 0., 0.] if letter == 'A' else [0., 1., 0.]


def score(args):
    queries = load(args.manifest)['queries']
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                dtype=torch.bfloat16).to('cuda').eval()
    tokens = [tokenizer.encode(letter, add_special_tokens=False) for letter in ('A', 'B', 'C')]
    if any(len(t) != 1 for t in tokens) or len({t[0] for t in tokens}) != 3:
        raise ValueError('Expected three distinct single-token choices')
    ids = [t[0] for t in tokens]
    domains = defaultdict(list)
    for r in queries:
        domains[r['domain']].append(r)
    result = []
    prompts_scored = 0
    non_ties = 0
    started = time.perf_counter()
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    for domain, rows in sorted(domains.items()):
        checkpoint = args.checkpoint / f'{domain}.json'
        model_files = [(p.name, p.stat().st_size) for p in sorted(args.model.iterdir()) if p.is_file()]
        instruction = QUOTE_INSTRUCTION if args.quote else INSTRUCTION
        inputs = [rows, instruction, args.batch_size, model_files]
        if args.quote:
            inputs.append(args.quote_tokens)
        digest = hashlib.sha256(json.dumps(inputs, sort_keys=True)
                                .encode()).hexdigest()
        if checkpoint.exists():
            saved = load(checkpoint)
            if saved['sha256'] != digest:
                raise ValueError(f'Checkpoint differs for {domain}')
            result.extend(saved['rows'])
            prompts_scored += saved['prompts']
            non_ties += saved.get('non_ties', 0)
            continue
        corpus = texts_for_domain(rows, args.data_root / domain / 'documents.jsonl')
        prompts = []
        shown = []
        for row in rows:
            query = bounded(tokenizer, row['query'], 1200)
            vocabulary = terms(row['query'])
            docs = {p[k] for p in row['proposals'] for k in ('incumbent', 'challenger')}
            excerpts = {d: bounded(tokenizer, evidence(corpus[d], vocabulary, 'dispersed'), 520) for d in docs}
            for p in row['proposals']:
                old, new = excerpts[p['incumbent']], excerpts[p['challenger']]
                for a, b in ((old, new), (new, old)):
                    prompt = f'{instruction}\n\nQuery\n{query}\n\nPassage A\n{a}\n\nPassage B\n{b}'
                    prompts.append(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                                tokenize=False, add_generation_prompt=True,
                                                                enable_thinking=False))
                    if args.quote:
                        shown.append((a, b))
        probs = []
        with torch.inference_mode():
            for i in range(0, len(prompts), args.batch_size):
                tokens = tokenizer(prompts[i:i+args.batch_size], add_special_tokens=False,
                                   padding=True, return_tensors='pt').to(model.device)
                if tokens.input_ids.shape[-1] > 4096:
                    raise ValueError(f'Oversized comparison for {domain}')
                if args.quote:
                    generated = model.generate(**tokens, max_new_tokens=args.quote_tokens, do_sample=False,
                                               temperature=None, top_p=None, top_k=None,
                                               pad_token_id=tokenizer.pad_token_id)
                    new_text = tokenizer.batch_decode(generated[:, tokens.input_ids.shape[-1]:],
                                                      skip_special_tokens=True)
                    probs.extend(grounded_choice(out, shown[i+j]) for j, out in enumerate(new_text))
                else:
                    scores = model(**tokens, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float()
                    probs.extend(scores.softmax(dim=-1).cpu().tolist())
        current = []
        offset = 0
        for r in rows:
            votes = []
            for _ in r['proposals']:
                forward, reverse = probs[offset:offset+2]
                votes.append([forward, [reverse[1], reverse[0], reverse[2]]])
                offset += 2
            current.append({k: r[k] for k in ('qid', 'domain')} | {'votes': votes})
        if offset != len(probs):
            raise ValueError(f'Votes differ from prepared comparisons for {domain}')
        active = sum(v[2] < .5 for v in probs) if args.quote else 0
        write_checkpoint(checkpoint, {'sha256': digest, 'rows': current, 'prompts': len(prompts),
                                      'non_ties': active})
        result.extend(current)
        prompts_scored += len(prompts)
        non_ties += active
        print(domain, len(result), '/', len(queries), 'queries;', prompts_scored, 'prompts;',
              round(time.perf_counter()-started), 'seconds', flush=True)
    return {'per_query': result, 'inference': {'model': args.model.name, 'prompts': prompts_scored,
                                             'mode': 'quote' if args.quote else 'logits', 'verified_choices': non_ties,
                                             'seconds': time.perf_counter()-started,
                                             'peak_mib': torch.cuda.max_memory_allocated()/2**20}}


def choose(proposals, votes, policy):
    kind, margin, agreement = policy
    best, chosen = 0., None
    for i, (p, v) in enumerate(zip(proposals, votes, strict=True)):
        if kind == 'none' or (kind != 'both' and p['type'] != kind):
            continue
        differences = [new-old for old, new, _ in v]
        mean = sum(differences)/2
        if mean <= margin or (agreement and any(new <= max(old, tie) for old, new, tie in v)):
            continue
        impact = 1/log2(p['slot']+2)
        if p['type'] == 'head':
            impact -= 1/log2(p['from_slot']+2)
        if mean*impact > best:
            best, chosen = mean*impact, i
    return chosen


def evaluate(args):
    queries = load(args.manifest)['queries']
    split, = {r['split'] for r in queries}
    keys = {(r['domain'], r['qid']) for r in queries}
    archives = [load(p) for p in args.scores]
    votes = index({'per_query': [r for archive in archives for r in archive['per_query']]})
    if votes.keys() != keys:
        raise ValueError('Score archive covers different query IDs')
    labels = {d: read_qrels(args.data_root / d / f'qrels_{split}.txt') for d, _ in keys}
    scored = []
    for row in queries:
        d, q = row['domain'], row['qid']
        active = votes[d, q]['votes']
        gold = labels[d][q]
        actions = {arm: choose(row['proposals'], active, policy) for arm, policy in POLICIES.items()}
        orders = {arm: row['reference'] if idx is None else row['proposals'][idx]['ranking']
                  for arm, idx in actions.items()}
        scored.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'actual_reference')} |
                      {'rankings': orders, 'scores': {a: ndcg10(v, gold) for a, v in orders.items()},
                       'actions': actions})
    if args.selection:
        if split != 'dev' or len(scored) != 519:
            raise ValueError('Expected all official dev queries')
        selected = load(args.selection)['selected']
        choices = {'dev': selected}
    else:
        if split != 'train' or len(scored) != 186:
            raise ValueError('Expected the 186 held-out training queries')
        def select(rows):
            return max(POLICIES, key=lambda a: macro((r['domain'], r['scores'][a]) for r in rows))
        selected = select(scored)
        choices = {f: select([r for r in scored if r['cohort'] != f]) for f in {r['cohort'] for r in scored}}

    def summary(rows):
        selected_names = [choices[r['cohort']] for r in rows]
        arms = {name: [name]*len(rows) for name in POLICIES} | {'selected': selected_names}
        return {name: {'score': macro((r['domain'], r['scores'][a])
                                      for r, a in zip(rows, actual, strict=True)),
                       'change': paired_delta([(r['domain'], r['scores'][a]-r['scores']['current'])
                                               for r, a in zip(rows, actual, strict=True)]),
                       'edits': sum(r['actions'][a] is not None for r, a in zip(rows, actual, strict=True))}
                for name, actual in arms.items()}

    result = {'selected': selected, 'choices': choices, 'per_query': scored,
              split: summary(scored), 'inference': [r['inference'] for r in archives]}
    if split == 'train':
        result['actual65'] = summary([r for r in scored if r['actual_reference']])
    elif abs(result[split]['current']['score'] - .3958942538410169) > 1e-12:
        raise ValueError('Official reference score changed')
    print(split, selected, choices, {a: (v['score'], v['edits']) for a, v in result[split].items()}, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    scoring, scoring_eval = sub.add_parser('score'), sub.add_parser('evaluate')
    for arg in (scoring, scoring_eval):
        for name in ('manifest', 'data-root', 'output'):
            arg.add_argument(f'--{name}', type=Path, required=True)
    scoring.add_argument('--model', type=Path, required=True)
    scoring.add_argument('--checkpoint', type=Path, required=True)
    scoring.add_argument('--batch-size', type=int, default=8)
    scoring.add_argument('--quote', action='store_true')
    scoring.add_argument('--quote-tokens', type=int, default=112)
    scoring_eval.add_argument('--scores', nargs='+', type=Path, required=True)
    scoring_eval.add_argument('--selection', type=Path)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('Provide a fresh output path')
    result = score(args) if args.action == 'score' else evaluate(args)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
