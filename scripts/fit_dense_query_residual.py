"""Fit a query-only retrieval correction using existing document embeddings."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from audit_anchor_dev import macro
from compare_anchor_reference import load
from coverage_cf_bound import ndcg10, read_qrels
from extend_evidence_adapter_pools import picks
from lib_reteco import load_examples, read_trec_run
from replay_anchor_edit_scope import paired_delta


PREFIX = 'Represent this sentence for searching relevant passages: '
STRENGTHS = {'original': 0., 'residual25': .25, 'residual50': .5, 'residual100': 1.}


def directories(root, split):
    return sorted(p for p in root.iterdir() if (p / f'examples_{split}.jsonl').is_file())


def encode_original(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from run_dense_generic import encode

    keys, questions = [], []
    for directory in directories(args.data_root, args.split):
        ids, queries = load_examples(directory, args.split)
        keys.extend((directory.name, qid) for qid in ids)
        questions.extend(PREFIX + q for q in queries)
    model = AutoModel.from_pretrained(args.encoder, local_files_only=True).to('cuda', dtype=torch.float16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True)
    vectors = encode(questions, model, tokenizer, 'cuda', 16, 512, 'mean', 'original queries')
    with args.output.open('xb') as file:
        np.savez_compressed(file, vectors=vectors, keys=np.asarray(keys), split=args.split)
    print(len(keys), 'query vectors written', flush=True)


def read_encoded(path, split):
    with np.load(path) as file:
        if file['split'].item() != split:
            raise ValueError('Query embeddings use a different split')
        keys = [tuple(key) for key in file['keys'].tolist()]
        vectors = file['vectors']
    if len(keys) != len(set(keys)) or vectors.shape != (len(keys), 1024) or not np.isfinite(vectors).all():
        raise ValueError('Unexpected original query vectors')
    return dict(zip(keys, vectors, strict=True))


def domain_index(path, embeddings):
    with (path / 'documents.jsonl').open(encoding='utf-8') as file:
        ids = [json.loads(line)['id'] for line in file if line.strip()]
    vectors = np.load(embeddings / f'doc_emb_mxbai_{len(ids)}.npy', mmap_mode='r')
    if vectors.shape != (len(ids), 1024) or len(ids) != len(set(ids)):
        raise ValueError(f'Unexpected document index for {path.name}')
    return ids, vectors


def fit(args, queries, holdout):
    records = []
    for path in directories(args.data_root, 'train'):
        d = path.name
        ids, documents = domain_index(path, args.embeddings)
        lookup = {doc: i for i, doc in enumerate(ids)}
        gold = read_qrels(path / 'qrels_train.txt')
        dense = read_trec_run(args.runs / f'dense_mxbai_{d}_train.trec')
        for q, relevant in gold.items():
            key = (d, q)
            if key in holdout or key not in queries:
                continue
            pos = [lookup[doc] for doc in sorted(relevant)]
            neg = [lookup[doc] for _, _, doc in dense[q][:100] if doc not in relevant][:24]
            if not pos or not neg:
                continue
            vector = queries[key]
            offset = documents[pos].mean(0) - documents[neg].mean(0)
            residual = offset - np.dot(vector, offset)*vector
            records.append((key, vector, residual))
        print(d, len(records), 'query correction targets', flush=True)
    weights = Counter(key[0] for key, _, _ in records)
    x, y = (np.asarray(part, dtype=np.float64) for part in zip(*[(q, v) for _, q, v in records]))
    w = np.array([1/np.sqrt(weights[key[0]]) for key, _, _ in records])
    w /= w.mean()
    query_mean, residual_mean = (np.average(v, axis=0, weights=w) for v in (x, y))
    xc, yc = x-query_mean, y-residual_mean
    coefficients = np.linalg.solve(xc.T @ (w[:, None]*xc) + .8*np.eye(1024), (xc*w[:, None]).T @ yc)
    return {'query_mean': query_mean, 'residual_mean': residual_mean, 'coefficients': coefficients,
            'domain_counts': dict(weights), 'fit_queries': len(records), 'ridge_lambda': .8}


def audit(args, projection, queries, rows, split):
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r['domain']].append(r)
    result, pools = [], {k: [] for k in ('existing60', *STRENGTHS)}
    for domain, slate in sorted(by_domain.items()):
        path = args.data_root / domain
        ids, docs = domain_index(path, args.embeddings)
        gold = read_qrels(path / f'qrels_{split}.txt')
        previous = read_trec_run(args.runs / f'dense_mxbai_{domain}_{split}.trec')
        original = np.stack([queries[domain, r['qid']] for r in slate])
        delta = projection['residual_mean'] + (original-projection['query_mean']) @ projection['coefficients']
        arrays = np.concatenate([original + alpha*delta for alpha in STRENGTHS.values()]).astype(np.float32)
        similarities = arrays @ docs.T
        indices = np.argpartition(-similarities, 119, axis=1)[:, :120]
        values = np.take_along_axis(similarities, indices, axis=1)
        indices = np.take_along_axis(indices, np.argsort(-values, axis=1, kind='stable'), axis=1)
        orders = np.asarray(ids)[indices].reshape(len(STRENGTHS), len(slate), 120)
        for j, row in enumerate(slate):
            qid, start = row['qid'], row['docs']
            if len(start) != 60 or len(set(start)) != 60:
                raise ValueError(f'Expected a fixed 60-document pool for {domain}/{qid}')
            old_top = {d for _, _, d in previous[qid][:100]}
            overlap = len(old_top & set(orders[0, j, :100]))/100
            current = {'existing60': start}
            current.update({name: start[:30] + picks(orders[i, j], set(start[:30]), 30)
                            for i, name in enumerate(STRENGTHS)})
            values = {}
            for name, candidates in current.items():
                relevant = set(candidates) & gold[qid]
                values[name] = ndcg10(sorted(relevant), gold[qid])
                pools[name].append(row | {'docs': candidates})
            result.append({'domain': domain, 'qid': qid, 'top100_agreement': overlap, 'oracles': values})
        print(domain, len(result), '/', len(rows), 'retrieval audits', flush=True)
    if np.mean([r['top100_agreement'] for r in result]) < .9:
        raise ValueError('Original query fails the existing mxbai retrieval check')
    summary = {name: {'oracle': macro((r['domain'], r['oracles'][name]) for r in result),
                      'change': paired_delta([(r['domain'], r['oracles'][name] - r['oracles']['existing60']) for r in result]),
                      'empty': sum(r['oracles'][name] == 0 for r in result)} for name in pools}
    return {'fit': {k: projection[k] for k in ('domain_counts', 'fit_queries', 'ridge_lambda')},
            'per_query': result, 'summary': summary}, pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    embedding, validation = (sub.add_parser(s) for s in ('encode', 'evaluate'))
    for p in (embedding, validation):
        p.add_argument('--data-root', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
    embedding.add_argument('--encoder', type=Path, required=True)
    embedding.add_argument('--split', choices=('train', 'dev'), required=True)
    for flag in ('embeddings', 'runs', 'vectors', 'holdout', 'pools'):
        validation.add_argument(f'--{flag}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    if args.action == 'encode':
        encode_original(args)
    else:
        if args.pools.exists():
            parser.error('Use a fresh pool output')
        queries = read_encoded(args.vectors, 'train')
        rows = load(args.holdout)['queries']
        keys = {(r['domain'], r['qid']) for r in rows}
        if len(keys) != len(rows) or len(rows) != 186 or {r['split'] for r in rows} != {'train'}:
            raise ValueError('Expected the fixed 186-query training holdout')
        projection = fit(args, queries, keys)
        report, pools = audit(args, projection, queries, rows, 'train')
        for path, value in ((args.output, report), (args.pools, pools)):
            with path.open('x', encoding='utf-8') as file:
                json.dump(value, file, separators=(',', ':'))
        print({k: (v['oracle'], v['empty']) for k, v in report['summary'].items()}, flush=True)


if __name__ == '__main__':
    main()
