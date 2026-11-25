"""Score the same evidence pools jointly in forward and reverse document order."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import terms
from compare_anchor_reference import load
from eval_anchor_reranker import write_checkpoint
from replay_anchor_edit_scope import z
from score_specialist_shortlists import TEMPORAL_INSTRUCT, evidence, select_shard


REVISION = 'e8a93f33f0b22108f8c2364f8484ce3422552fbc'
CODE_SHA256 = 'ec6612461b4307eb3bab089e6916c00881b7e6b2e8edfbd56a9ace4560244837'


def score(args):
    if hashlib.sha256((args.model / 'modeling.py').read_bytes()).hexdigest() != CODE_SHA256:
        raise ValueError('The local Jina scorer differs from the reviewed snapshot')
    queries = load(args.manifest)['queries']
    if args.shards > 1:
        queries = select_shard(queries, args.shards, args.shard_index)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, trust_remote_code=True,
                                      dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    model._tokenizer = tokenizer
    domains = defaultdict(list)
    for row in queries:
        domains[row['domain']].append(row)
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    completed, prompt_count = [], 0
    started = time.perf_counter()
    for domain, rows in sorted(domains.items()):
        path = args.checkpoint / f'{domain}.json'
        fingerprint = hashlib.sha256(json.dumps([REVISION, TEMPORAL_INSTRUCT, rows], sort_keys=True)
                                    .encode()).hexdigest()
        if path.exists():
            checkpoint = load(path)
            if checkpoint['fingerprint'] != fingerprint:
                raise ValueError(f'Checkpoint inputs differ for {domain}')
            completed.extend(checkpoint['rows'])
            continue
        texts = read_documents(args.data_root / domain / 'documents.jsonl', {d for r in rows for d in r['docs']})
        scored = []
        with torch.inference_mode():
            for row in rows:
                docs = row['docs']
                query = row['query']
                vocabulary = terms(query)
                excerpts = [evidence(texts[d], vocabulary, 'dispersed') for d in docs]
                first = model._compute_single_batch(query, excerpts, instruction=TEMPORAL_INSTRUCT)
                forward = first.scores.reshape(-1).float().cpu().numpy()
                del first
                second = model._compute_single_batch(query, excerpts[::-1], instruction=TEMPORAL_INSTRUCT)
                reverse = second.scores.reshape(-1).flip(0).float().cpu().numpy()
                del second
                if forward.size != len(docs) or reverse.size != len(docs):
                    raise ValueError(f'Unexpected listwise scores for {domain}/{row["qid"]}')
                values = .5 * (z(forward) + z(reverse))
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite listwise scores for {domain}/{row["qid"]}')
                keys = ('qid', 'domain', 'split', 'cohort')
                scored.append({key: row[key] for key in keys} | {
                    'scores': dict(zip(docs, values.tolist(), strict=True)),
                    'order_scores': {'forward': forward.tolist(), 'reverse': reverse.tolist()},
                })
                prompt_count += 2
        write_checkpoint(path, {'fingerprint': fingerprint, 'rows': scored})
        completed.extend(scored)
        print(domain, len(completed), '/', len(queries), 'queries', round(time.perf_counter()-started), flush=True)
    return {'model': 'jina-reranker-v3.5', 'model_revision': REVISION, 'evidence_mode': 'dispersed',
            'instruction': TEMPORAL_INSTRUCT, 'per_query': completed,
            'score_layout': 'mean of z-scaled forward and reversed whole-list similarities',
            'inference': {'prompts_computed': prompt_count, 'total_prompts': 2*len(completed),
                          'peak_mib': torch.cuda.max_memory_allocated()/2**20,
                          'seconds': time.perf_counter()-started}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ('data-root', 'manifest', 'model', 'checkpoint', 'output'):
        ap.add_argument(f'--{flag}', type=Path, required=True)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    args = ap.parse_args()
    if not 0 <= args.shard_index < args.shards:
        ap.error('Shard index must be within the range of shards')
    if args.output.exists():
        ap.error('Use a fresh output path')
    result = score(args)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, separators=(',', ':'))
        file.write('\n')


if __name__ == '__main__':
    main()
