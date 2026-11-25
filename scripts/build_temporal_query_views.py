"""Generate two evidence requests and encode them with the existing mxbai encoder."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from compare_anchor_reference import load
from lib_reteco import load_examples
from run_dense_generic import encode


PREFIX = 'Represent this sentence for searching relevant passages: '
INSTRUCTION = (
    'Write two complementary stand-alone search queries for finding documents that answer the question below. '
    'Each search query should target a different evidence need. Keep the question’s specific names, versions, '
    'and time restrictions relevant to that need. Use words a matching document is likely to contain. '
    'Never give an answer or invent names, dates, relevant facts, or additional requirements. '
    'Return exactly two lines beginning with A: and B: respectively. Keep each line under 35 words.'
)
EXTRACTIVE_INSTRUCTION = (
    'Select existing parts of the question to form two complementary evidence searches. '
    'For each search select up to three numbered parts that belong together. Keep explicit time conditions '
    'with the requirement they qualify. Select a different emphasis for each search. Return exactly two '
    'lines using part numbers only, for example A: 1,3 and B: 2,4. Copy the numbers as provided.'
)


def fragments(query):
    parts = re.split(r'(?<=[.?!;])\s+|\n+', query)
    return [part.strip() for part in parts if part.strip()]


def interpret(query, completion, extractive):
    lines = re.findall(r'^\s*([AB]):\s*(.+\S)\s*$', completion, flags=re.MULTILINE)
    if [label for label, _ in lines] != ['A', 'B']:
        return [query]*2, False
    raw = [s.strip() for _, s in lines]
    if not extractive:
        return (raw, True) if all(4 <= len(s.split()) <= 45 for s in raw) else ([query]*2, False)
    sections = fragments(query)
    if not all(re.fullmatch(r'\d+(?:\s*,\s*\d+){0,2}', s) for s in raw):
        return [query]*2, False
    indices = [sorted({1, *(int(i) for i in s.split(','))}) for s in raw]
    if any(i < 1 or i > len(sections) for group in indices for i in group):
        return [query]*2, False
    return [' '.join(sections[i-1] for i in group) for group in indices], True


def generate(rows, model_path, batch_size, instruction, extractive):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                dtype=torch.bfloat16).to('cuda').eval()
    views = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            group = rows[i:i+batch_size]
            prompts = []
            for row in group:
                query = row['query']
                question = '\n'.join(f'{j}. {line}' for j, line in enumerate(fragments(query), 1)) if extractive else query
                prompts.append(tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': f'{instruction}\n\nQuestion\n{question}'}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            inputs = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=False).to(model.device)
            answers = model.generate(**inputs, max_new_tokens=140, do_sample=False,
                                     temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
            completions = tokenizer.batch_decode(answers[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for row, completion in zip(group, completions, strict=True):
                parts, parsed = interpret(row['query'], completion, extractive)
                views.append({'domain': row['domain'], 'qid': row['qid'], 'parts': parts,
                              'completion': completion, 'parsed': parsed})
            if i == 0 or len(views) % (batch_size*10) == 0 or len(views) == len(rows):
                print(len(views), '/', len(rows), 'query views generated', flush=True)
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-root', 'rewriter', 'encoder', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--extractive', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    rows = load(args.manifest)['queries']
    split, = {r['split'] for r in rows}
    queries = {d: dict(zip(*load_examples(args.data_root / d, split), strict=True))
               for d in {r['domain'] for r in rows}}
    instruction = EXTRACTIVE_INSTRUCTION if args.extractive else INSTRUCTION
    views = generate(rows, args.rewriter, args.batch_size, instruction, args.extractive)
    torch.cuda.empty_cache()
    model = AutoModel.from_pretrained(args.encoder, local_files_only=True).to('cuda', dtype=torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True)
    texts = [PREFIX + t for row in views for t in (queries[row['domain']][row['qid']], *row['parts'])]
    vectors = encode(texts, model, tokenizer, 'cuda', 16, 512, 'mean', 'query views')
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, vectors=vectors.reshape(len(views), 3, vectors.shape[-1]),
                            metadata=json.dumps({'split': split, 'rewriter': args.rewriter.name,
                                                 'encoder': args.encoder.name, 'instruction': instruction,
                                                 'views': views}, ensure_ascii=False))
    print(sum(r['parsed'] for r in views), '/', len(views), 'parsed successfully', flush=True)


if __name__ == '__main__':
    main()
