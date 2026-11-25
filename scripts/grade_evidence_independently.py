"""Regrade the same excerpts with a fresh relevance rubric and optional evidence notes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from replay_anchor_fallback import index
from score_specialist_shortlists import evidence


RUBRIC = (
    'Grade the document as evidence for the question below. Identify which concrete part of the '
    'question the excerpt can answer. Assess the entity, time relation, or version for that part. '
    'Treat any evidence notes as a lead. Confirm the claim using the document excerpt itself. '
    'An unrelated condition from another part of the question is irrelevant to the claim. '
    'A date mentioned in passing, a compatible example, or a discussion of the subject is weak '
    'evidence unless it answers a concrete part. Give 3 for a clear answer or decisive evidence '
    'for one or more parts of the question, 2 for information useful toward solving a specific '
    'part, 1 for related background, and 0 for material with no usable connection or a direct '
    'contradiction of a required time relation. Output only the grade digit.'
)
GRADE = re.compile(r'(?i)\b(?:rating|grade)\s*:\s*[0-3]\b')


def prompt(tokenizer, query, passage, notes):
    body = f'{RUBRIC}\n\nQuestion\n{query}\n\nDocument excerpt\n{passage}'
    if notes:
        body += f'\n\nCandidate evidence notes\n{notes}'
    return tokenizer.apply_chat_template([{'role': 'user', 'content': body}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def assess(args):
    manifest = index({'per_query': load(args.manifest)['queries']})
    current = load(args.scores)
    rows = index(current)
    if manifest.keys() != rows.keys():
        raise ValueError('The earlier judgements and new rubric require the same queries')
    groups = defaultdict(list)
    for key, query in manifest.items():
        obs = rows[key]['observations']
        if obs.keys() != set(query['docs']):
            raise ValueError(f'Expected identical candidates for {key}')
        groups[key[0]].append((query, obs))

    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                dtype=torch.bfloat16).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in ('0', '1', '2', '3')]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 4:
        raise ValueError('Expected four distinct grade tokens')
    tokens = [x[0] for x in ids]
    grades = torch.arange(4, device=model.device, dtype=torch.float32)
    started = time.perf_counter()
    result = []
    with torch.inference_mode():
        for domain, group in sorted(groups.items()):
            docs = {doc for query, _ in group for doc in query['docs']}
            texts = read_documents(args.data_root / domain / 'documents.jsonl', docs)
            prompts, entries = [], []
            for query, obs in group:
                topic = query['query']
                vocabulary = terms(topic)
                for doc in query['docs']:
                    extracted = evidence(texts[doc], vocabulary, 'dispersed')
                    notes = GRADE.split(obs[doc]['response'], maxsplit=1)[0].strip()
                    prompts.extend((prompt(tokenizer, topic, extracted, ''),
                                    prompt(tokenizer, topic, extracted, notes)))
                    entries.append(obs[doc])
            predicted = []
            for i in range(0, len(prompts), args.batch_size):
                batch = tokenizer(prompts[i:i+args.batch_size], padding=True, add_special_tokens=False,
                                  return_tensors='pt').to(model.device)
                if batch.input_ids.shape[-1] > 4096:
                    raise ValueError(f'Rubric context exceeds 4096 tokens in {domain}')
                logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, tokens].float()
                predicted.extend((logits.softmax(-1) @ grades).cpu().tolist())
            for item, direct, reasoned in zip(entries, predicted[::2], predicted[1::2], strict=True):
                item.update({'direct': direct, 'reasoned': reasoned})
            result.extend(rows[query['domain'], query['qid']] for query, _ in group)
            print(domain, len(result), '/', len(rows), 'queries', round(time.perf_counter()-started), flush=True)
    return {'per_query': result, 'model': args.model.name, 'instruction': RUBRIC,
            'inference': {'seconds': time.perf_counter()-started,
                          'pairs': sum(len(r['observations']) for r in result),
                          'peak_mib': torch.cuda.max_memory_allocated()/2**20}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'scores', 'data-root', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    result = assess(args)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
