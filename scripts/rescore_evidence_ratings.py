"""Replay ordinal grading after removing explicit ratings from generated explanations."""
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
from eval_evidence_reasoner import INSTRUCTION
from replay_anchor_fallback import index
from score_specialist_shortlists import evidence


RATING = re.compile(r'\bRating\s*:\s*[0-3]\b', flags=re.IGNORECASE)


def assess(args):
    manifest = index({'per_query': load(args.manifest)['queries']})
    record = load(args.scores)
    rows = index(record)
    if rows.keys() != manifest.keys() or record['model'] != args.model.name or record['instruction'] != INSTRUCTION:
        raise ValueError('The original observations must match the manifest and grader')
    work = defaultdict(list)
    for key, row in rows.items():
        if row['observations'].keys() != set(manifest[key]['docs']):
            raise ValueError(f'The original candidates differ for {key}')
        for doc, result in row['observations'].items():
            response = result['response']
            expected = RATING.search(response, response.find('Why:')+1)
            if expected:
                work[key[0]].append((key, doc, response[:expected.start()].rstrip(), result))
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                dtype=torch.bfloat16).to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in ('0', '1', '2', '3')]
    if not all(len(x) == 1 for x in ids) or len({x[0] for x in ids}) != 4:
        raise ValueError('Expected four distinct rating tokens')
    tokens = [x[0] for x in ids]
    grades = torch.arange(4, device=model.device, dtype=torch.float32)
    started = time.perf_counter()
    count = 0
    with torch.inference_mode():
        for domain, entries in sorted(work.items()):
            texts = read_documents(args.data_root / domain / 'documents.jsonl', {doc for _, doc, _, _ in entries})
            prompts = []
            for key, doc, response, _ in entries:
                query = manifest[key]['query']
                passage = evidence(texts[doc], terms(query), 'dispersed')
                content = f'{INSTRUCTION}\n\nQuestion\n{query}\n\nDocument\n{passage}'
                first = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prompts.append(first + response + '\nRating: ')
            for i in range(0, len(prompts), args.batch_size):
                enc = tokenizer(prompts[i:i+args.batch_size], padding=True, add_special_tokens=False,
                                return_tensors='pt').to(model.device)
                logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, tokens].float()
                for value, (_, _, _, result) in zip((logits.softmax(-1) @ grades).cpu().tolist(),
                                                    entries[i:i+args.batch_size], strict=True):
                    result['reasoned_before_mask'] = result['reasoned']
                    result['reasoned'] = value
                    count += 1
            print(domain, count, 'ratings masked', round(time.perf_counter()-started), flush=True)
    record['inference']['masked_ratings'] = count
    record['inference']['mask_seconds'] = time.perf_counter()-started
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'scores', 'data-root', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    result = assess(args)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))


if __name__ == '__main__':
    main()
