"""Extract evidence before grading documents from a fixed candidate pool."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from audit_anchor_dev import macro
from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import passages, terms
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from eval_anchor_reranker import write_checkpoint
from eval_qwen_reranker_adapter import candidate_judges, choose, grade, metric, summarize
from eval_rerank_headroom_study import diagnose
from replay_anchor_edit_scope import z
from replay_anchor_fallback import index
from score_specialist_shortlists import evidence


INSTRUCTION = (
    'Judge whether this document contains credible evidence for a requirement of the question. '
    'Time, entities, and version conditions must match the requirement it answers. A related topic alone is weak evidence. '
    'Use the following rating scale for your assessment: 0 means unrelated or explicit time conflict; 1 means '
    'topically related but lacks an answer; 2 means usable partial evidence; 3 means a clear answer to a specific requirement. '
    'First write two short lines, Quote: followed by one verbatim supporting excerpt or NONE, then Why: '
    'followed by a brief explanation of what the excerpt supports. Keep both lines to at most 45 words total.'
)


def prepare(args):
    source = load(args.manifest)['queries']
    report = load(args.repair)
    split, = {r['split'] for r in source}
    refs = index({'per_query': report['oof_per_query' if split == 'train' else 'dev_per_query']})
    judges = [candidate_judges(files)[0]['pretrained'] for files in (args.first, args.second)]
    groups = []
    for row in source:
        key = row['domain'], row['qid']
        all_docs = row['docs']
        if len(all_docs) != 60 or len(set(all_docs)) != 60:
            raise ValueError(f'Expected an expanded pool for {key}')
        if any(set(all_docs) != score[key]['scores'].keys() for score in judges):
            raise ValueError(f'Expected the two matched score sources for {key}')
        policy = report['choices'][row['cohort']] if split == 'train' else report['selected']
        current = refs[key]['rankings'][policy]
        proposed = []
        for tier in (all_docs[:30], all_docs[30:]):
            allowed = [d for d in tier if d not in current]
            score = np.mean([z([judge[key]['scores'][d] for d in allowed]) for judge in judges], axis=0)
            proposed.append([allowed[i] for i in np.argsort(-score, kind='stable')[:args.per_tier]])
        near, deep = proposed
        docs = current + near[:2] + deep[:2] + near[2:] + deep[2:]
        if len(docs) != len(set(docs)) or len(docs) != 10+2*args.per_tier:
            raise ValueError(f'Unexpected candidate pool for {key}')
        groups.append(row | {'docs': docs})
    return {'queries': groups}


def verified(response, excerpt):
    quote = re.search(r'(?:^|\n)\s*Quote:\s*([^\r\n]+)', response, flags=re.IGNORECASE)
    if not quote:
        return False
    wanted = re.findall(r'\b\w+\b', quote.group(1).casefold())
    words = re.findall(r'\b\w+\b', excerpt.casefold())
    return len(wanted) >= 5 and f' {" ".join(wanted)} ' in f' {" ".join(words)} '


def clause_excerpts(body, query, vocabulary):
    by_clause = [terms(part) for part in re.split(r'(?<=[.?!;])\s+', query)]
    by_clause = [part for part in by_clause if len(part) >= 3]
    if len(by_clause) < 2:
        return evidence(body, vocabulary, 'dispersed')
    options = passages(body, vocabulary)
    first = options[0]
    covered = terms(first)

    def complement(window):
        observed = terms(window)
        return max(len((observed - covered) & part) * len(observed & part) / len(part)
                   for part in by_clause)

    second = max(options, key=complement)
    return first + '\n[Additional excerpt]\n' + second if complement(second) else first


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    source = load(args.manifest)['queries']
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                local_files_only=True).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    choices = [tokenizer.encode(d, add_special_tokens=False) for d in ('0', '1', '2', '3')]
    if any(len(ids) != 1 for ids in choices) or len({i[0] for i in choices}) != 4:
        raise ValueError('Expected four distinct rating tokens')
    ids = [i[0] for i in choices]
    tiers = torch.arange(4, device=model.device, dtype=torch.float32)
    groups = defaultdict(list)
    for row in source:
        groups[row['domain']].append(row)
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    model_files = [(p.name, p.stat().st_size) for p in sorted(args.model.iterdir()) if p.is_file()]
    output, started = [], time.perf_counter()
    for domain, rows in sorted(groups.items()):
        path = args.checkpoint / f'{domain}.json'
        config = [rows, INSTRUCTION, args.batch_size, args.max_new_tokens, model_files]
        if args.evidence_mode != 'dispersed':
            config.append(args.evidence_mode)
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if path.exists():
            saved = load(path)
            if saved['fingerprint'] != fingerprint:
                raise ValueError(f'Changed reasoner inputs for {domain}')
            output.extend(saved['rows'])
            continue
        texts = read_documents(args.data_root / domain / 'documents.jsonl', {d for row in rows for d in row['docs']})
        prompts, excerpts = [], []
        for row in rows:
            query = row['query']
            vocabulary = terms(query)
            for d in row['docs']:
                passage = (clause_excerpts(texts[d], query, vocabulary) if args.evidence_mode == 'clause'
                           else evidence(texts[d], vocabulary, 'dispersed'))
                content = f'{INSTRUCTION}\n\nQuestion\n{query}\n\nDocument\n{passage}'
                prompts.append(tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                            tokenize=False, add_generation_prompt=True,
                                                            enable_thinking=False))
                excerpts.append(passage)
        observations = []
        with torch.inference_mode():
            for start in range(0, len(prompts), args.batch_size):
                texts = prompts[start:start+args.batch_size]
                initial = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
                if initial.input_ids.shape[-1] + args.max_new_tokens + 8 > 4096:
                    raise ValueError(f'Evidence assessment exceeds 4096 tokens in {domain}')
                direct_tokens = tokenizer([p + 'Rating: ' for p in texts], padding=True, add_special_tokens=False,
                                          return_tensors='pt').to(model.device)
                direct_logits = model(**direct_tokens, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float()
                direct = (direct_logits.softmax(-1) @ tiers).cpu().tolist()
                generated = model.generate(**initial, max_new_tokens=args.max_new_tokens, do_sample=False,
                                           top_k=None, top_p=None, temperature=None, pad_token_id=tokenizer.pad_token_id)
                replies = tokenizer.batch_decode(generated[:, initial.input_ids.shape[-1]:], skip_special_tokens=True)
                judged = tokenizer([p + r + '\nRating: ' for p, r in zip(texts, replies, strict=True)],
                                   padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
                raw = model(**judged, use_cache=False, logits_to_keep=1).logits[:, -1, ids].float()
                grades = (raw.softmax(-1) @ tiers).cpu().tolist()
                batch_excerpts = excerpts[start:start+args.batch_size]
                for text, initial_grade, grade, passage in zip(replies, direct, grades, batch_excerpts, strict=True):
                    observations.append({'direct': initial_grade, 'reasoned': grade,
                                         'verified': verified(text, passage), 'response': text})
        scored = []
        offset = 0
        for row in rows:
            end = offset + len(row['docs'])
            items = observations[offset:end]
            kv = ('qid', 'domain', 'split', 'cohort')
            scored.append({k: row[k] for k in kv} | {'observations': dict(zip(row['docs'], items, strict=True))})
            offset = end
        if offset != len(observations):
            raise ValueError(f'Unexpected number of observations for {domain}')
        write_checkpoint(path, {'fingerprint': fingerprint, 'rows': scored})
        output.extend(scored)
        print(domain, len(output), '/', len(source), 'queries', round(time.perf_counter()-started), flush=True)
    return {'per_query': output, 'model': args.model.name, 'instruction': INSTRUCTION,
            'evidence_mode': args.evidence_mode,
            'inference': {'seconds': time.perf_counter()-started,
                          'pairs': sum(len(r['observations']) for r in output),
                          'peak_mib': torch.cuda.max_memory_allocated()/2**20}}


def one_edit(docs, observations, use_reasoning, require_quote):
    feature = 'reasoned' if use_reasoning else 'direct'
    discount = 1 / np.log2(2 + np.arange(10))
    candidates = [(0, rank) for rank in range(1, 5)]
    candidates.extend((i, j) for i in range(7, 10) for j in range(10, len(docs)))
    proposed, best = docs[:10], 0.
    for i, j in candidates:
        incumbent, contender = observations[docs[i]], observations[docs[j]]
        margin = contender[feature] - incumbent[feature]
        if margin < .65 or contender[feature] < 1.8 or (require_quote and not contender['verified']):
            continue
        impact = discount[i] - (discount[j] if j < 10 else 0.)
        if impact * margin > best:
            best = impact * margin
            proposed = docs[:10]
            proposed[i] = docs[j]
            if j < 10:
                proposed[j] = docs[i]
    return proposed


def shared_prefix_scale(values):
    values = np.asarray(values)
    prefix = values[:14]
    return (values - prefix.mean()) / max(float(prefix.std()), 1e-8)


def evaluate(args):
    source = load(args.manifest)['queries']
    archives = [load(p) for p in args.scores]
    identities = {(archive['model'], archive['instruction'], archive.get('evidence_mode', 'dispersed'))
                  for archive in archives}
    if len(identities) != 1:
        raise ValueError('Reasoner model and scoring prompt must match across archives')
    measured = {}
    for archive in archives:
        for r in archive['per_query']:
            key = r['domain'], r['qid']
            if key in measured:
                previous = measured[key]
                if ((r['split'], r['cohort']) != (previous['split'], previous['cohort'])
                        or not r['observations'].keys().isdisjoint(previous['observations'])):
                    raise ValueError(f'Overlapping document scores or split mismatch for {key}')
                previous['observations'].update(r['observations'])
            else:
                measured[key] = r
    keys = {(r['domain'], r['qid']) for r in source}
    split, = {r['split'] for r in source}
    if keys != measured.keys() or len(source) != (186 if split == 'train' else 519):
        raise ValueError('Reasoner scores and evaluated population disagree')
    gold = {d: read_qrels(args.data_root / d / f'qrels_{split}.txt') for d, _ in keys}
    rows, ceilings = [], []
    for row in source:
        d, q, docs = row['domain'], row['qid'], row['docs']
        obs = measured[d, q]['observations']
        if set(docs) != obs.keys():
            raise ValueError('Observed evidence documents differ from the manifest')
        discount = 1 / np.log2(2 + np.arange(len(docs)))
        prior = z(discount)
        rankings = {'current': docs[:10]}
        for name in ('direct', 'reasoned', 'verified'):
            if name == 'verified':
                raw = [obs[x]['reasoned'] if obs[x]['verified'] else obs[x]['direct'] for x in docs]
            else:
                raw = [obs[x][name] for x in docs]
            values = z(raw)
            for alpha in (.3, .5):
                order = [docs[i] for i in np.argsort(-((1-alpha)*prior+alpha*values), kind='stable')]
                rankings[f'{name}/{alpha:g}'] = order[:10]
                rankings[f'{name}/{alpha:g}_keep_first'] = [docs[0], *[x for x in order if x != docs[0]][:9]]
            if len(docs) > 14:
                combined = .5*shared_prefix_scale(discount) + .5*shared_prefix_scale(raw)
                rankings[f'{name}/0.5_common14'] = [docs[i] for i in np.argsort(-combined, kind='stable')[:10]]
        for name, reasoned, quote in (('direct', False, False), ('reasoned', True, False), ('verified', True, True)):
            rankings[f'{name}/single_edit'] = one_edit(docs, obs, reasoned, quote)
        rows.append({k: row[k] for k in ('domain', 'qid', 'cohort', 'split', 'actual_reference')} |
                    {'rankings': rankings})
        ceilings.append((d, ndcg10(sorted(set(docs) & gold[d][q]), gold[d][q])))
    rows = grade(rows, split, args.data_root)
    arms = tuple(rows[0]['scores'])
    if split == 'train':
        selected = choose(rows, arms)
        by_fold = {f: choose([r for r in rows if r['cohort'] != f], arms) for f in {r['cohort'] for r in rows}}
        result = {'selected': selected, 'by_fold': by_fold, 'training': summarize(rows, selected, by_fold),
                  'original_65': summarize([r for r in rows if r['actual_reference']], selected, by_fold)}
    else:
        selected = load(args.selection)['selected']
        result = {'selected': selected, 'dev': summarize(rows, selected),
                  'diagnostics': diagnose(rows, selected, args.data_root, 'current')}
        if abs(result['dev']['current']['score'] - .3958942538410169) > 1e-12:
            raise ValueError('The official reference score changed')
    result['oracle'] = macro(ceilings)
    result['inference'] = [archive['inference'] for archive in archives]
    result['verified_choices'] = sum(x['verified'] for row in measured.values() for x in row['observations'].values())
    result['per_query'] = rows
    print(split, selected, 'macro', metric(rows, [selected]*len(rows)), 'ceiling', result['oracle'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    setup, scoring, evaluation = (commands.add_parser(k) for k in ('prepare', 'score', 'evaluate'))
    for command in (setup, scoring, evaluation):
        for name in ('manifest', 'output'):
            command.add_argument(f'--{name}', type=Path, required=True)
    for name in ('first', 'second'):
        setup.add_argument(f'--{name}', type=Path, nargs='+', required=True)
    setup.add_argument('--repair', type=Path, required=True)
    setup.add_argument('--per-tier', type=int, choices=(2, 6), default=2)
    for name in ('model', 'checkpoint', 'data-root'):
        scoring.add_argument(f'--{name}', type=Path, required=True)
    scoring.add_argument('--batch-size', type=int, default=8)
    scoring.add_argument('--max-new-tokens', type=int, default=80)
    scoring.add_argument('--evidence-mode', choices=('dispersed', 'clause'), default='dispersed')
    evaluation.add_argument('--scores', type=Path, nargs='+', required=True)
    evaluation.add_argument('--selection', type=Path)
    evaluation.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    run = {'prepare': prepare, 'score': score, 'evaluate': evaluate}[args.action]
    result = run(args)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
