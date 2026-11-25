"""Score the fixed 34-document abstention pools with Qwen3-Reranker."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_anchor_prompt_outliers import trim_words
from benchmark_pairwise_replacement_v2 import passages, strip_html, terms
from compare_anchor_reference import load
from eval_anchor_reranker import write_checkpoint
from replay_evidence_shortlist import inputs, query_text
from run_qwen3_rerank import INSTRUCT, PREFIX, SUFFIX


TEMPORAL_INSTRUCT = (
    'Retrieve documents that give usable evidence for the query. A document can support one part of '
    'the query. For the part it supports, respect explicit time periods and ordering. '
    'Matching a topic or mentioning an unrelated date is insufficient.'
)


def evidence(body, vocabulary, mode):
    if mode == 'expanded':
        words = strip_html(body).split()
        if len(words) <= 768:
            return ' '.join(words)
        return passages(body, vocabulary, win=768, stride=384)[0]
    options = passages(body, vocabulary)
    first = options[0]
    if mode == 'dispersed':
        missing = vocabulary - terms(first)
        if missing:
            second = max(options, key=lambda text: len(terms(text) & missing))
            if terms(second) & missing:
                first = first + '\n[Additional excerpt]\n' + second
    return trim_words(first, 2048, 256)


def focus_part(query, passage, policy):
    clauses = [part.strip() for part in re.split(r'(?<=[.?!;])\s+', query) if len(terms(part)) >= 3]
    if not clauses:
        return query
    if policy == 'fixed':
        return clauses[0]
    observed = terms(passage)
    return max(clauses, key=lambda clause: len(terms(clause) & observed) / len(terms(clause)) ** .5)


def prepare(root, args):
    result = []
    for split, paths, frozen, scope in (
        ('train', args.oof, args.oof_frozen, args.oof_scope),
        ('dev', [args.dev], args.dev_frozen, args.dev_scope)
    ):
        texts = query_text(root, split)
        for r in inputs(paths, frozen, load(scope), split):
            outsiders = [d for d in r['docs'] if d not in r['start']]
            keep = set(r['start'] + outsiders[:24])
            docs = [d for d in r['docs'] if d in keep]
            if len(docs) != 34:
                raise ValueError(f'Incomplete fixed pool for {r["qid"]}')
            result.append({k: r[k] for k in ('domain', 'qid', 'cohort')} | {
                'split': split, 'docs': docs, 'query': texts[r['domain'], r['qid']],
            })
    return sorted(result, key=lambda r: (r['domain'], r['split'], r['qid']))


def score(manifest, data_root, model_path, batch_size, checkpoint, mode='reference', adapter=None, focus=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                dtype=torch.bfloat16).to('cuda').eval()
    identity = None
    if adapter:
        from qwen_reranker_adapter import load_adapter

        identity = {'sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    'training': load_adapter(model, adapter)}
    yes, no = [tokenizer.encode(label, add_special_tokens=False) for label in ('yes', 'no')]
    if len(yes) != 1 or len(no) != 1 or yes == no:
        raise ValueError('Expected distinct one-token decisions')
    prefix, suffix = (tokenizer.encode(text, add_special_tokens=False) for text in (PREFIX, SUFFIX))
    body_budget = 4096 - len(prefix) - len(suffix)
    rows = defaultdict(list)
    for row in manifest:
        rows[row['domain']].append(row)
    checkpoint.mkdir(parents=True, exist_ok=True)
    signature = [(p.name, p.stat().st_size) for p in sorted(model_path.iterdir()) if p.is_file()]
    stats = {'tokens': 0, 'prompts': 0, 'resumed_domains': 0}
    instruction = INSTRUCT if mode == 'reference' else TEMPORAL_INSTRUCT
    if focus:
        instruction += (' Judge whether the evidence directly answers the highlighted part while '
                        'preserving the entities and time constraints in the full query.')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    completed = []
    for domain, group in sorted(rows.items()):
        path = checkpoint / f'{domain}.json'
        fingerprint_inputs = [group, signature, instruction, PREFIX, SUFFIX, batch_size]
        if mode != 'reference':
            fingerprint_inputs.append(mode)
        if identity:
            fingerprint_inputs.append(identity['sha256'])
        if focus:
            fingerprint_inputs.append(focus)
        fingerprint = hashlib.sha256(json.dumps(fingerprint_inputs, sort_keys=True).encode()).hexdigest()
        if path.exists():
            saved = load(path)
            if saved['fingerprint'] != fingerprint:
                raise ValueError(f'Resume inputs differ for {domain}')
            completed.extend(saved['scores'])
            stats['tokens'] += saved['tokens']
            stats['prompts'] += sum(len(r['docs']) for r in group)
            stats['resumed_domains'] += 1
            continue
        needed = {doc for r in group for doc in r['docs']}
        corpus = {}
        with (data_root / domain / 'documents.jsonl').open(encoding='utf-8') as file:
            for line in file:
                document = json.loads(line)
                if document['id'] in needed:
                    corpus[document['id']] = document['content']
                    if len(corpus) == len(needed):
                        break
        if corpus.keys() != needed:
            raise ValueError(f'Incomplete candidate texts in {domain}')
        prompts = []
        for r in group:
            query = r['query']
            vocabulary = terms(query)
            for doc in r['docs']:
                passage = evidence(corpus[doc], vocabulary, mode)
                guide = ''
                if focus:
                    part = tokenizer.encode(focus_part(query, passage, focus), add_special_tokens=False)[:96]
                    guide = f'\n<Highlighted part>: {tokenizer.decode(part)}'
                if mode == 'expanded':
                    prompt_start = f'<Instruct>: {instruction}\n<Query>: {query}{guide}\n<Document>: '
                    header = tokenizer.encode(prompt_start, add_special_tokens=False)
                    if len(header) >= body_budget:
                        raise ValueError(f'Query leaves no expanded evidence budget in {domain}')
                    document = tokenizer.encode(passage, add_special_tokens=False)[:body_budget - len(header)]
                    prompts.append(header + document)
                else:
                    prompts.append(f'<Instruct>: {instruction}\n<Query>: {query}{guide}\n<Document>: {passage}')
        values, tokens = [], 0
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                items = prompts[start:start + batch_size]
                body = items if mode == 'expanded' else tokenizer(items, add_special_tokens=False)['input_ids']
                if any(len(b) > body_budget for b in body):
                    raise ValueError(f'An input in {domain} exceeds the 4096-token budget')
                batch = tokenizer.pad({'input_ids': [prefix + b + suffix for b in body]},
                                      padding=True, return_tensors='pt').to(model.device)
                logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                values.extend((logits[:, yes[0]] - logits[:, no[0]]).cpu().tolist())
                tokens += int(batch['attention_mask'].sum().item())
        scores = []
        offset = 0
        for r in group:
            end = offset + len(r['docs'])
            scores.append({key: r[key] for key in ('domain', 'qid', 'split', 'cohort')} |
                          {'scores': dict(zip(r['docs'], values[offset:end], strict=True))})
            offset = end
        if offset != len(values):
            raise ValueError(f'Score count differs for {domain}')
        write_checkpoint(path, {'fingerprint': fingerprint, 'scores': scores, 'tokens': tokens})
        completed.extend(scores)
        stats['tokens'] += tokens
        stats['prompts'] += len(prompts)
        print(domain, len(completed), '/', len(manifest), 'queries', round(time.perf_counter()-started), flush=True)
    report = {'model': model_path.name, 'evidence_mode': mode, 'instruction': instruction, 'inference': stats | {
        'seconds': time.perf_counter() - started, 'peak_mib': torch.cuda.max_memory_allocated() / 2**20,
    }, 'per_query': completed}
    if identity:
        report['adapter'] = identity
    if focus:
        report['focus'] = focus
    return report


def select_shard(queries, count, slot):
    groups = defaultdict(list)
    for row in queries:
        groups[row['domain']].append(row)
    if count > len(groups):
        raise ValueError('The number of shards exceeds the number of domains')
    shards = [[] for _ in range(count)]
    load_by_shard = [0] * count
    ordered = sorted(groups.items(), key=lambda item: (-sum(len(r['docs']) for r in item[1]), item[0]))
    for _, group in ordered:
        target = min(range(count), key=load_by_shard.__getitem__)
        shards[target].extend(group)
        load_by_shard[target] += sum(len(r['docs']) for r in group)
    return shards[slot]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--model-revision')
    parser.add_argument('--focus', choices=('aligned', 'fixed'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--evidence-mode', choices=('reference', 'temporal', 'dispersed', 'expanded'),
                        default='reference')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--oof', type=Path, nargs=2)
    parser.add_argument('--dev', type=Path)
    for name in ('oof-frozen', 'dev-frozen', 'oof-scope', 'dev-scope'):
        parser.add_argument(f'--{name}', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shards:
        parser.error('Shard index must be within the range of shards')
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.prepare:
        report = {'queries': prepare(args.root, args)}
        print(len(report['queries']), 'query pools prepared', flush=True)
    else:
        queries = load(args.manifest)['queries']
        if args.shards > 1:
            queries = select_shard(queries, args.shards, args.shard_index)
        report = score(queries, args.data_root or args.root / 'data/track1_tempo',
                       args.model, args.batch_size, args.checkpoint, args.evidence_mode, args.adapter, args.focus)
        report['shard'] = {'index': args.shard_index, 'count': args.shards}
        if args.model_revision:
            report['model_revision'] = args.model_revision
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
