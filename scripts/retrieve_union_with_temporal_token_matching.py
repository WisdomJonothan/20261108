"""Compare dense and token-level matching from the same BGE-M3 checkpoint."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from time import perf_counter

from audit_rejected_swap_evidence import read_documents
from benchmark_pairwise_replacement_v2 import strip_html
from compare_anchor_reference import load
from lib_reteco import load_examples, read_trec_run
from replay_anchor_fallback import index
from retrieve_union_with_reasoning_encoder import four_new, union_for


REVISION = 'f11a3f18ef0f9740f5268d633d025d11b33e1e82'
SOURCES = ('m3_dense', 'm3_maxsim', 'm3_temporal_maxsim')
TEMPORAL_CUE = re.compile(
    r'\b(?:(?:1\d|20)\d{2}s?|before|after|since|until|throughout|during|prior to|as of|no later than)\b',
    re.IGNORECASE,
)
LIMIT = 512


def time_weights(question, offsets):
    spans = [m.span() for m in TEMPORAL_CUE.finditer(question)]
    return [1 + int(a < b and any(a < end and b > start for start, end in spans))
            for a, b in offsets]


def representation(tokenizer, model, projection, texts, with_offsets=False):
    import torch

    batch = tokenizer(texts, padding=True, truncation=True, max_length=LIMIT,
                      return_offsets_mapping=with_offsets, return_tensors='pt')
    offsets = batch.pop('offset_mapping', None)
    batch = batch.to('cuda')
    states = model(**batch).last_hidden_state
    dense = torch.nn.functional.normalize(states[:, 0], dim=-1)
    contextual = torch.nn.functional.normalize(projection(states[:, 1:]), dim=-1)
    mask = batch.attention_mask[:, 1:].bool()
    return dense, contextual, mask, offsets


def encode_passages(passages, tokenizer, model, projection, batch_size, label):
    import torch

    order = sorted(range(len(passages)), key=lambda i: -len(passages[i]))
    device = model.device
    dense = torch.empty((len(passages), model.config.hidden_size), device=device, dtype=model.dtype)
    features = torch.zeros((len(passages), LIMIT-1, projection.out_features),
                           device=device, dtype=model.dtype)
    masks = torch.zeros((len(passages), LIMIT-1), device=device, dtype=torch.bool)
    start = perf_counter()
    for offset in range(0, len(passages), batch_size):
        indices = order[offset:offset+batch_size]
        vectors, tokens, mask, _ = representation(
            tokenizer, model, projection, [passages[i] for i in indices])
        width = tokens.shape[1]
        dense[indices] = vectors
        features[indices, :width] = tokens
        masks[indices, :width] = mask
        if offset % (20 * batch_size) == 0:
            print(label, offset + len(indices), '/', len(passages), round(perf_counter()-start), flush=True)
    return dense, features, masks


def score_candidates(query, indices, vectors, tokenizer, model, projection):
    import torch

    docs_dense, docs_tokens, docs_mask = vectors
    dense, query_tokens, mask, offsets = representation(tokenizer, model, projection, [query], True)
    count = int(mask[0].sum())
    query_tokens = query_tokens[0, :count].T
    weights = torch.tensor(time_weights(query, offsets[0, 1:1+count].tolist()),
                           dtype=torch.float32, device=model.device)
    collection = []
    for start in range(0,len(indices),24):
        block = indices[start:start+24]
        similarities = docs_tokens[block] @ query_tokens
        similarities.masked_fill_(~docs_mask[block, :, None], float('-inf'))
        maxima = similarities.max(dim=1).values.float()
        pooled = (docs_dense[block] @ dense[0]).float()
        collection.append(torch.stack((pooled, maxima.mean(-1), (maxima @ weights) / weights.sum()), dim=-1))
    scores = torch.cat(collection)
    if not torch.isfinite(scores).all():
        raise ValueError('Expected finite scores for every candidate')
    return scores.cpu().numpy(), int((weights > 1).sum())


def score(args):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='right')
    model = AutoModel.from_pretrained(args.model, local_files_only=True,
                                      dtype=torch.float16, use_safetensors=True).to('cuda').eval()
    weights = torch.load(args.model/'colbert_linear.pt', map_location='cpu', weights_only=True)
    projection = torch.nn.Linear(model.config.hidden_size, weights['weight'].shape[0])
    projection.load_state_dict(weights)
    projection.to(device=model.device, dtype=model.dtype).eval()

    reference = index(load(args.reference))
    by_domain = defaultdict(list)
    for row in load(args.manifest)['queries'][args.shard_index::args.shards]:
        by_domain[row['domain'], row['split']].append(row)
    output = []
    with torch.inference_mode():
        for (domain, split), rows in sorted(by_domain.items()):
            first, second = (read_trec_run(root/f'dense_{tag}_{domain}_{split}.trec')
                             for root, tag in ((args.dense_runs, 'mxbai'), (args.bge_runs, 'bge')))
            groups = {r['qid']: union_for(r, first, second, args.depth)[1] for r in rows}
            ids = list(dict.fromkeys(d for pool in groups.values() for d in pool))
            positions = {doc:i for i,doc in enumerate(ids)}
            texts = read_documents(args.data_root/domain/'documents.jsonl', set(ids))
            passages = [(strip_html(texts[doc][:51200]) or ' ') for doc in ids]
            vectors = encode_passages(passages, tokenizer, model, projection, args.batch_size, domain)
            queries = dict(zip(*load_examples(args.data_root/domain, split), strict=True))
            for row in rows:
                qid = row['qid']
                pool = groups[qid]
                scores, cues = score_candidates(queries[qid], [positions[d] for d in pool], vectors,
                                                tokenizer, model, projection)
                order = np.argsort(-scores, axis=0, kind='stable')
                old = set(row['docs'])
                offers = {name:four_new((pool[i] for i in order[:, arm]), old)
                          for arm, name in enumerate(SOURCES)}
                output.append({k:row[k] for k in ('domain','qid','split','cohort','actual_reference')} |
                              {'original':row['docs'], 'prior_ten':reference[domain, qid]['rankings']['verified_0.5'],
                               'offers':offers, 'union_count':len(pool), 'temporal_token_count':cues})
            del vectors
            print(domain, len(output), 'queries complete', flush=True)
    return {'models':{'m3_revision':REVISION, 'source_depth':args.depth, 'token_limit':LIMIT},
            'sources':list(SOURCES), 'per_query':output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','reference','model','data-root','dense-runs','bge-runs','output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    for name, value in (('depth',200),('batch-size',24),('shards',1),('shard-index',0)):
        parser.add_argument(f'--{name}', type=int, default=value)
    args = parser.parse_args()
    if args.output.exists() or not (args.depth >= 64 and 0 <= args.shard_index < args.shards and args.batch_size > 0):
        parser.error('Use a fresh output and check source depth, batch size and shard index')
    result = score(args)
    with args.output.open('x',encoding='utf8') as f:
        json.dump(result,f,separators=(',',':'))


if __name__ == '__main__':
    main()
